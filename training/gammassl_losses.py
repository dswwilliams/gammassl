import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
from utils.candr_utils import crop_by_box_and_resize
from utils.training_metric_utils import calculate_consistency2certainty_prob_metrics, calculate_p_certain_per_class, get_consistency_metrics
from utils.validation_utils import calculate_miou
import sys
sys.path.append("../")
from models.mask2former_loss import SetCriterion
from models.matcher import HungarianMatcher


class Sharpener(nn.Module):
    """
    Reduces the entropy of the distribution.
    Raises the probabilities to the power of (1/temperature) and then re-normalises.
    Lower temperature -> lower entropy (for temp <  1)
    """
    def __init__(self, temperature=0.25):
        super(Sharpener, self).__init__()
        self.temperature = temperature
    
    def forward(self, p, dim=1):
        sharp_p = p**(1/self.temperature)
        sharp_p /= torch.sum(sharp_p, dim=dim, keepdim=True)
        return sharp_p


class SoftProbCrossEntropy(nn.Module):
    """
    Cross-entropy loss function for which the target is a soft probability distribution.
    """

    def __init__(self, dim, reduction="mean"):
        super(SoftProbCrossEntropy, self).__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, target_probs, query_probs):
        """ 
        xent = sum( - p * log(q) ) = sum*(log(q**-p))
        where: p = target and q = input
        """

        # for numerical stability
        target_probs = target_probs + 1e-7
        query_probs = query_probs + 1e-7

        xent = torch.log(query_probs**-target_probs)
            
        xent = torch.sum(xent, dim=self.dim, keepdim=False)
        if self.reduction == "mean":
            return xent.mean()
        elif self.reduction == "sum":
            return xent.sum()
        elif self.reduction == "none":
            return xent



def uniformity_loss_fn(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class GammaSSLLosses:
    """
    Implements loss functions for the different types of GammaSSL training.
    """
    def __init__(self, opt, num_known_classes):
        self.opt = opt
        self.num_known_classes = num_known_classes

        self.temperature = opt.temperature

        # determine whether to include void class in loss
        ignore_index = -100 if self.opt.include_void else self.num_known_classes

        self.sharpen = Sharpener(temp=opt.sharpen_temp)
        self.hard_xent = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.soft_xent = SoftProbCrossEntropy(dim=1, reduction="none")
        self.m2f_criterion = SetCriterion(
                        num_classes=num_known_classes,
                        matcher=HungarianMatcher(num_points=self.opt.num_points),
                        weight_dict=None,       # doesn't appear to be used
                        eos_coef=0.1,       # correct from paper
                        losses=['labels', 'masks'],
                        num_points=self.opt.num_points,
                        oversample_ratio=3,             # from config
                        importance_sample_ratio=0.75,   # from config
                    )

    
    def calculate_consistency_loss(self, seg_masks_t, seg_masks_q, gamma_masks_q):
        """
        Calculates the GammaSSL consistency loss.
        The consistency is maximised only where seg_masks_q is certain.
        Consistency is measured as the cross-entropy between seg_masks_t and seg_masks_q.

        Args:
            seg_masks_t (tensor): Segmentation masks for target branch [bs, K, H, W]
            seg_masks_q (tensor): Segmentation masks for query branch [bs, K, H, W]
            gamma_masks_q (tensor): Masks defining pixels where seg_masks_q is uncertain [bs, H, W] 

        Returns:
            loss_c (tensor): consistency loss [1]
            consistency_metrics (dict): dictionary of training metrics
        """
        _sharpen = self.sharpen if self.opt.sharpen_temp is not None else lambda x: x


        logits_t = seg_masks_t / self.temperature
        logits_q = seg_masks_q / self.temperature
        xent = self.soft_xent(_sharpen(torch.softmax(logits_t, dim=1)), torch.softmax(logits_q, dim=1))


        
        # certainty mask is 1 where the query branch is certain else 0
        certainty_mask = (1 - gamma_masks_q.float())
        if self.opt.no_filtering:
            certainty_mask = torch.ones_like(certainty_mask)

        if certainty_mask.sum() == 0:
            # avoid division by zero
            loss_c = torch.zeros_like(xent, requires_grad=True)[0,0,0]
        else:
            # calculating mean cross-entropy over the regions that are certain
            loss_c = (xent * certainty_mask).sum() / certainty_mask.sum()

        consistency_metrics = get_consistency_metrics(logits_t, logits_q, certainty_mask, detailed_metrics=self.opt.detailed_metrics)

        return loss_c, consistency_metrics
    

    
    def calculate_masking_ssl_loss(self, seg_masks_t, seg_masks_q, gamma_masks_q, raw_masks_q, raw_masks_t, valid_masking_region_masks=None, return_loss_imgs=False):
        ### calculating consistency loss ###
        # where certain, minimise cross-entropy over ALL K classes 
        ssl_losses = {}
        ssl_metrics = {}

        if self.opt.sharpen_temp is not None:
            _sharpen = self.sharpen
        else:
            _sharpen = lambda x: x

        target_temp = self.opt.loss_c_temp
        query_temp = self.opt.loss_c_temp
        logits_1_to_K_t = seg_masks_t / target_temp
        logits_1_to_K_q = seg_masks_q / query_temp

        p_y_given_x_t = _sharpen(torch.softmax(logits_1_to_K_t, dim=1))
        p_y_given_x_q = torch.softmax(logits_1_to_K_q, dim=1)
        xent = self.soft_xent(p_y_given_x_t, p_y_given_x_q)

        ssl_metrics["mean_max_softmax_query"] = torch.max(p_y_given_x_q, dim=1)[0].mean().cpu()
        ssl_metrics["mean_max_softmax_target"] = torch.max(p_y_given_x_t, dim=1)[0].mean().cpu()


        with torch.no_grad():
            H, W = logits_1_to_K_q.shape[-2:]                
            if raw_masks_q is not None:
                raw_masks_q = raw_masks_q.reshape(-1, H//14, W//14)
                raw_masks_q = F.interpolate(raw_masks_q.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1).long()
            else:
                raw_masks_q = torch.ones(logits_1_to_K_q.shape[0],H,W).to(self.device).long()

            ms_imgs_q = torch.max(p_y_given_x_q.detach(), dim=1)[0]
            ms_imgs_t = torch.max(p_y_given_x_t.detach(), dim=1)[0]

            ssl_metrics["mean_max_softmax_query_unmasked"] = torch.div(
                                                                (ms_imgs_q * (1-raw_masks_q.float())).sum(), 
                                                                (1-raw_masks_q.float()).sum(),
                                                                ).cpu()
            
            ssl_metrics["mean_max_softmax_target_unmasked"] = torch.div(
                                                                (ms_imgs_t * (1-raw_masks_q.float())).sum(), 
                                                                (1-raw_masks_q.float()).sum(),
                                                                ).cpu()
            
            ssl_metrics["mean_max_softmax_query_masked"] = torch.div(
                                                                (ms_imgs_q * (raw_masks_q.float())).sum(), 
                                                                (raw_masks_q.float()).sum(),
                                                                ).cpu()
            
            ssl_metrics["mean_max_softmax_target_masked"] = torch.div(
                                                                (ms_imgs_t * (raw_masks_q.float())).sum(), 
                                                                (raw_masks_q.float()).sum(),
                                                                ).cpu()
        


        
        ### mask certain regions ###
        certainty_mask = (1 - gamma_masks_q.float())
        # patch is masked if masks_i = 1, else is not masked
        global_mask = certainty_mask

        if valid_masking_region_masks is not None:
            # only compute loss for regions that could have been masked, i.e. are uncertain according to the target
            global_mask = global_mask * valid_masking_region_masks.float()


        if return_loss_imgs:
            xent_loss_imgs = xent.detach() * global_mask.detach()
        else:
            xent_loss_imgs = None

        if self.opt.no_filtering:
            global_mask = torch.ones_like(global_mask)
        if global_mask.sum() == 0:
            xent = torch.zeros_like(xent, requires_grad=True)[0,0,0]
        else:
            xent = (xent * global_mask).sum() / global_mask.sum()
        loss_c = xent

        ### calculating metrics ###
        with torch.no_grad():
            segs_t = torch.argmax(seg_masks_t, dim=1).cpu()
            segs_q = torch.argmax(seg_masks_q, dim=1).cpu()

            ssl_metrics["p_certain_q"] = certainty_mask.mean().cpu()
            ssl_metrics["p_certain_q_unmasked"] = ((certainty_mask * (1-raw_masks_q.float())).sum()/ (1-raw_masks_q.float()).sum()).cpu()
            ssl_metrics["p_certain_q_masked"] = ((certainty_mask * (raw_masks_q.float())).sum()/ (raw_masks_q.float()).sum()).cpu()

            ssl_metrics["p_certain_per_class_q"], ssl_metrics["p_certain_per_class_t"] = calculate_p_certain_per_class(
                                                                                                                certainty_mask, 
                                                                                                                segs_q, 
                                                                                                                segs_t, 
                                                                                                                num_known_classes=self.num_known_classes,
                                                                                                                )

            # of the pixels that are consistent, calculate which classes they belong to
            consistency_masks = torch.eq(segs_q, segs_t).float().cpu()
            p_consistent_per_class = torch.zeros(self.num_known_classes).cpu()
            for k in range(self.num_known_classes):
                p_consistent_per_class[k]= torch.eq((consistency_masks.cpu() * (segs_q.cpu() + 1) - 1), k).float().sum() / consistency_masks.sum()
            
            consistency_masks = consistency_masks.to(self.device)
            certainty_mask = certainty_mask.to(self.device)
            ssl_metrics["p_consistent_per_class"] = p_consistent_per_class
            ssl_metrics["p_consistent"] = consistency_masks.float().mean().cpu()
            # only calculate p_consistent_unmasked on the unmasked regions
            ssl_metrics["p_consistent_unmasked"] = ((consistency_masks * (1-raw_masks_q.float())).sum()/ (1-raw_masks_q.float()).sum()).cpu()
            ssl_metrics["p_consistent_masked"] = ((consistency_masks * (raw_masks_q.float())).sum()/ (raw_masks_q.float()).sum()).cpu()


            ssl_metrics["mean_max_softmax_query_masked_consistent"] = (ms_imgs_q * consistency_masks * raw_masks_q.float()).sum() / (consistency_masks * raw_masks_q.float()).sum()
            ssl_metrics["mean_max_softmax_query_masked_inconsistent"] = (ms_imgs_q * (1-consistency_masks) * raw_masks_q.float()).sum() / ((1-consistency_masks) * raw_masks_q.float()).sum()
            ssl_metrics["mean_max_softmax_query_unmasked_consistent"] = (ms_imgs_q * consistency_masks * (1-raw_masks_q.float())).sum() / (consistency_masks * (1-raw_masks_q.float())).sum()
            ssl_metrics["mean_max_softmax_query_unmasked_inconsistent"] = (ms_imgs_q * (1-consistency_masks) * (1-raw_masks_q.float())).sum() / ((1-consistency_masks) * (1-raw_masks_q.float())).sum()


            consistency_masks = consistency_masks.cpu()
            certainty_mask = certainty_mask.cpu()

            certainty_metrics = calculate_consistency2certainty_prob_metrics(
                                                    accuracy_masks=consistency_masks,
                                                    confidence_masks=certainty_mask.cpu(),
                                                    )

            for key in certainty_metrics:
                if not (key == "p_certain"):
                    ssl_metrics[key] = (certainty_metrics[key].nansum() / (~certainty_metrics[key].isnan()).sum()).cpu()
        
        if return_loss_imgs:
            return xent_loss_imgs
        else:
            return loss_c, ssl_metrics
    

    
    def calculate_uniformity_loss(self, features, projection_net=None, rbf_t=2):
        ### uniformity loss ###
        # NOTE: average pooled THEN projected
        low_res_features = F.avg_pool2d(
                                    features,
                                    kernel_size=(self.opt.uniformity_kernel_size, self.opt.uniformity_kernel_size),
                                    stride=self.opt.uniformity_stride)
        if projection_net is not None:
            low_res_features = projection_net(low_res_features)
        bs, feature_dim, h, w = low_res_features.shape
        uniform_loss = uniformity_loss_fn(F.normalize(low_res_features.permute(0,2,3,1).reshape(bs*h*w, feature_dim), dim=1, p=2), 
                                            t=rbf_t, 
                                            no_exp=False)
        return uniform_loss
    

    
    @staticmethod
    def calculate_prototype_loss(prototypes, output_metrics=False):
        # Dot product of normalized prototypes is cosine similarity.
        product = torch.matmul(prototypes, prototypes.t()) + 1
        if output_metrics:
            with torch.no_grad():
                sep = (product - torch.diag(torch.diag(product)) - 1).max()
        # Remove diagonal from loss.
        product -=  2*torch.diag(torch.diag(product))
        # Minimize maximum cosine similarity.
        loss = product.max(dim=1)[0]

        if output_metrics:
            return loss.mean(), sep
        else:
            return loss.mean()
    

    
    def calculate_sup_loss(self, labelled_seg_masks, labels, labelled_crop_boxes_A):
        sup_metrics = {}

        ### supervised loss ###
        labels = crop_by_box_and_resize(labels.unsqueeze(1).float(), labelled_crop_boxes_A, mode="nearest").squeeze(1).long()
        # NOTE: not divided by self.opt.temperature
        sup_loss = self.hard_xent(labelled_seg_masks, labels)
        sup_loss = sup_loss.mean()


        with torch.no_grad():
            ### supervised metrics ###
            labelled_segs = torch.argmax(labelled_seg_masks, dim=1)
            # the supervised loss function is calculated over void class as well -> include in metric
            sup_metrics["labelled_miou"] = calculate_miou(labelled_segs, labels, num_classes=self.num_known_classes+1)
            sup_metrics["labelled_accuracy"] = torch.eq(labelled_segs, labels).float().mean()
        return sup_loss, sup_metrics
    

    
    def calculate_masking_sup_loss(self, seg_masks, labels, masks):
        masking_metrics = {}

        masking_loss = self.hard_xent(seg_masks, labels)
        # This loss is only calculated over the masked regions
        masking_loss = (masking_loss * masks).sum() / masks.sum()

        with torch.no_grad():
            ### supervised metrics ###
            labelled_segs = torch.argmax(seg_masks, dim=1)
            # the supervised loss function is calculated over void class as well -> include in metric
            masking_metrics["labelled_miou"] = calculate_miou(labelled_segs, labels, num_classes=self.num_known_classes+1)
            masking_metrics["labelled_accuracy"] = torch.eq(labelled_segs, labels).float().mean()

        return masking_loss, masking_metrics
    

    @staticmethod
    def semantic_inference(mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    
    def calculate_m2f_losses(self, outputs, labels, labelled_crop_boxes_A, masking_masks=None):
        """
        - outputs["pred_logits"]: [B, Q, K+1]
        - outputs["pred_masks"]: [B, Q, H, W]
        - labels: [B, H, W]
        """
        metrics = {}
        labels = crop_by_box_and_resize(labels.unsqueeze(1).float(), labelled_crop_boxes_A, mode="nearest").squeeze(1).long()

        bs = labels.shape[0]
        K = self.num_known_classes

        masks = F.one_hot(labels, num_classes=K+1).permute(0, 3, 1, 2)[:,:K,:,:].long()
        class_labels = torch.arange(K).unsqueeze(0).expand(bs, -1)
        targets = []    
        for batch_no in range(bs):
            target = {}
            target["labels"] = class_labels[batch_no]       # [K]
            target["masks"] = masks[batch_no]               # [K, H, W]
            targets.append(target)
        
        losses = self.m2f_criterion(outputs, targets)
        

        loss_ce = losses["loss_ce"] * 2 if "loss_ce" in losses else None
        loss_dice = losses["loss_dice"] * 5 if "loss_dice" in losses else None
        loss_mask = losses["loss_mask"] * 5 if "loss_mask" in losses else None
        del losses

        with torch.no_grad():
            seg_masks = self.semantic_inference(mask_cls=outputs["pred_logits"], mask_pred=outputs["pred_masks"])
            segs = torch.argmax(seg_masks, dim=1)
            metrics["m2f_miou"] = calculate_miou(segs, labels, num_classes=self.num_known_classes)
            metrics["m2f_accuracy"] = torch.eq(segs, labels).float().mean()
            if masking_masks is not None:
                print(f"masking_masks.shape: {masking_masks.shape}")
                masking_masks = masking_masks.reshape(-1, segs.shape[-2]//14, segs.shape[-1]//14)
                print(f"masking_masks.shape: {masking_masks.shape}")
                new_dims = (segs.shape[-2], segs.shape[-1])
                print(f"new_dims: {new_dims}")
                masking_masks = F.interpolate(masking_masks.unsqueeze(1).float(), size=(segs.shape[-2], segs.shape[-1]), mode="nearest").squeeze(1)
                print(f"masking_masks.shape: {masking_masks.shape}")
                print(f"segs.shape: {segs.shape}")
                print(f"labels.shape: {labels.shape}")
                metrics["m2f_accuracy_on_masked"] = (torch.eq(segs, labels).float() * masking_masks).sum() / masking_masks.sum()
                metrics["m2f_accuracy_on_unmasked"] = (torch.eq(segs, labels).float() * (1-masking_masks)).sum() / (1-masking_masks).sum()

            metrics["sup_mean_max_softmax"] = torch.softmax(seg_masks, dim=1).max(1)[0].mean().cpu()
            # metrics["sup_mean_max_softmax_temp"] = torch.softmax(seg_masks/self.opt.temperature, dim=1).max(1)[0].mean().cpu()

        return loss_ce, loss_dice, loss_mask, metrics