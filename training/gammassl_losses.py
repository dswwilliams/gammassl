import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
from utils.crop_utils import crop_by_box_and_resize
from utils.training_metric_utils import calculate_consistency2certainty_prob_metrics, calculate_p_certain_per_class
from utils.test_metric_utils import calculate_miou
import sys
sys.path.append("../")
from models.mask2former_loss import SetCriterion



def sharpen(p, dim=1, temp=0.25):
    sharp_p = p**(1./temp)
    sharp_p /= torch.sum(sharp_p, dim=dim, keepdim=True)
    return sharp_p


class Sharpener(nn.Module):
    def __init__(self, temp):
        super(Sharpener, self).__init__()
        self.temp = temp
    
    def forward(self, x):
        return sharpen(x, dim=1, temp=self.temp)



class TrueCrossEntropy(nn.Module):
    def __init__(self, dim, reduction="mean", class_weights=None, void_class_id=19, gamma_class_weight=1):
        super(TrueCrossEntropy, self).__init__()
        self.dim = dim
        self.reduction = reduction


        self.class_weights = torch.ones(void_class_id+1)
        self.class_weights[void_class_id] = gamma_class_weight
        self.class_weights = self.class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        print("TrueCrossEntropy class weights", self.class_weights.flatten())

    def forward(self, target_probs, query_probs):
        """ 
        xent = sum( - p * log(q) ) = sum*(log(q**-p))
        p: target
        q: input
        """
        
        p = target_probs
        q = query_probs

        p = p + 1e-7
        q = q + 1e-7

        xent = torch.log(q**-p)
        if self.class_weights is not None:
            xent = xent * self.class_weights[:,:xent.shape[1],:,:].to(xent.device)
            
        xent = torch.sum(xent, dim=self.dim, keepdim=False)
        if self.reduction == "mean":
            return xent.mean()
        elif self.reduction == "sum":
            return xent.sum()
        elif self.reduction == "none":
            return xent


def uniformity_loss_fn(x, t=2, no_exp=False):
    if no_exp:
        return torch.pdist(x, p=2).pow(2).mul(-t).mean()
    else:
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class GammaSSLLosses:
    def __init__(self, opt, device, num_known_classes):
        self.opt = opt
        self.num_known_classes = num_known_classes
        self.device = device

        self.sharpen = Sharpener(temp=opt.sharpen_temp)

        ### loading in class weights ###
        if not self.opt.sunrgbd:                # dont have class weights for sunrgbd -> can't use class weights
            if self.opt.use_class_weights_sup or self.opt.use_class_weights_ssl:
                if self.opt.use_cpu:
                    with open(os.path.join("/Users/dw/data/class_weights/cityscapes_class_weights_19c.pkl"), "rb") as file:
                        class_weights = np.array(pickle.load(file))
                        class_weights = torch.from_numpy(class_weights).float().to(self.device)
                else:
                    with open(os.path.join(self.opt.cityscapes_class_weights_path), "rb") as file:
                        class_weights = np.array(pickle.load(file))
                        class_weights = torch.from_numpy(class_weights).float().to(self.device)
                class_weights = class_weights/class_weights.min()
                class_weights = (class_weights - class_weights.min())/2 + class_weights.min()
            else:
                class_weights = torch.ones(self.num_known_classes).to(self.device)
        else:
            class_weights = torch.ones(self.num_known_classes).to(self.device)

        ### setting up loss functions ###
        if self.opt.include_void:
            ignore_index = -100             # i.e. void class is included in supervised loss
            # add class weight for void class
            class_weights = torch.cat((class_weights, class_weights.mean()*torch.ones(1, device=class_weights.device)), dim=0)
        else:
            # ignore void pixel labels
            ignore_index = self.num_known_classes
        print("class_weights: ", class_weights)

        # define supervised loss function
        if self.opt.use_class_weights_sup:
            self.sup_xent_loss_fn = nn.CrossEntropyLoss(reduction="none", weight=class_weights, ignore_index=ignore_index)
        else:
            self.sup_xent_loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

        # define self-supervised loss function
        if self.opt.use_class_weights_ssl:
            self.ssl_xent_loss_fn = TrueCrossEntropy(dim=1, reduction="none", class_weights=class_weights, void_class_id=self.num_known_classes)
        else:
            self.ssl_xent_loss_fn = TrueCrossEntropy(dim=1, reduction="none", void_class_id=self.num_known_classes)

        from models.matcher import HungarianMatcher
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

    ##########################################################################################################################################################
    def calculate_ssl_loss(self, seg_masks_t, seg_masks_q, gamma_masks_q):
        ### calculating consistency loss ###
        # where certain, minimise cross-entropy over ALL K classes 
        ssl_losses = {}
        ssl_metrics = {}

        if self.opt.sharpen_temp is not None:
            _sharpen = self.sharpen
        else:
            _sharpen = lambda x: x

        target_temp = self.opt.temperature
        query_temp = self.opt.temperature
        logits_1_to_K_t = seg_masks_t / target_temp
        logits_1_to_K_q = seg_masks_q / query_temp
        xent = self.ssl_xent_loss_fn(_sharpen(torch.softmax(logits_1_to_K_t, dim=1)), torch.softmax(logits_1_to_K_q, dim=1))

        ssl_metrics["mean_max_softmax_query"] = torch.max(torch.softmax(logits_1_to_K_q.detach(), dim=1), dim=1)[0].mean().cpu()
        ssl_metrics["mean_max_softmax_target"] = torch.max(torch.softmax(logits_1_to_K_t.detach(), dim=1), dim=1)[0].mean().cpu()
        
        ### mask certain regions ###
        certainty_mask = (1 - gamma_masks_q.float())
        if self.opt.no_filtering:
            certainty_mask = torch.ones_like(certainty_mask)
        if certainty_mask.sum() == 0:
            xent = torch.zeros_like(xent, requires_grad=True)[0,0,0]
        else:
            xent = (xent * certainty_mask).sum() / certainty_mask.sum() 
        loss_c = xent


        ### calculating metrics ###
        with torch.no_grad():
            segs_t = torch.argmax(seg_masks_t, dim=1).cpu()
            segs_q = torch.argmax(seg_masks_q, dim=1).cpu()

            ssl_metrics["p_certain_q"] = certainty_mask.mean().cpu()

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
            
            ssl_metrics["p_consistent_per_class"] = p_consistent_per_class
            ssl_metrics["p_consistent"] = consistency_masks.float().mean().cpu()

            certainty_metrics = calculate_consistency2certainty_prob_metrics(
                                                    accuracy_masks=consistency_masks,
                                                    confidence_masks=certainty_mask.cpu(),
                                                    )

            for key in certainty_metrics:
                if not (key == "p_certain"):
                    ssl_metrics[key] = (certainty_metrics[key].nansum() / (~certainty_metrics[key].isnan()).sum()).cpu()

        return loss_c, ssl_metrics
    ##########################################################################################################################################################

    ##########################################################################################################################################################
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
        xent = self.ssl_xent_loss_fn(p_y_given_x_t, p_y_given_x_q)

        ssl_metrics["mean_max_softmax_query"] = torch.max(p_y_given_x_q, dim=1)[0].mean().cpu()
        ssl_metrics["mean_max_softmax_target"] = torch.max(p_y_given_x_t, dim=1)[0].mean().cpu()

        with torch.no_grad():
            H, W = logits_1_to_K_q.shape[-2:]                
            raw_masks_q = raw_masks_q.reshape(-1, H//14, W//14)
            raw_masks_q = F.interpolate(raw_masks_q.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1).long()

            ms_imgs_q = torch.max(p_y_given_x_q.detach(), dim=1)[0]
            print(f"in calculate_masking_ssl_loss, ms_imgs_q min mean max: {ms_imgs_q.min()}, {ms_imgs_q.mean()}, {ms_imgs_q.max()}")
            ms_imgs_t = torch.max(p_y_given_x_t.detach(), dim=1)[0]
            print(f"in calculate_masking_ssl_loss, ms_imgs_t min mean max: {ms_imgs_t.min()}, {ms_imgs_t.mean()}, {ms_imgs_t.max()}")

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
        if self.opt.loss_c_unmasked_only:
            # only calculate the loss where raw_masks_q = 0
            global_mask = (certainty_mask * (1-raw_masks_q.float()))
        else:
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

            # NOTE: that if p_certain = p_consistent, then certainty_mask.sum() = consistency_masks.sum(), 
            # therefore p_consistent_g_certain = p_certain_g_consistent, and p_consistent_g_uncertain = p_certain_g_inconsistent
            # ssl_metrics["p_certain_g_consistent"] = ((certainty_mask * consistency_masks).sum() / (consistency_masks.sum() + 1e-8)).cpu()
            # ssl_metrics["p_certain_g_inconsistent"] = ((certainty_mask * (1-consistency_masks)).sum() / ((1-consistency_masks).sum() + 1e-8)).cpu()
            # ssl_metrics["p_uncertain_g_consistent"] = (((1-certainty_mask) * consistency_masks).sum() / (consistency_masks.sum() + 1e-8)).cpu()
            # ssl_metrics["p_uncertain_g_inconsistent"] = (((1-certainty_mask) * (1-consistency_masks)).sum() / ((1-consistency_masks).sum() + 1e-8)).cpu()

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
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def calculate_kl_uniform_loss(self, seg_masks_q, gamma_masks_q):
        ### calculating consistency loss ###
        # where certain, minimise cross-entropy over ALL K classes 
        
        kld_metrics = {}

        if self.opt.sharpen_temp is not None:
            _sharpen = self.sharpen
        else:
            _sharpen = lambda x: x

        logits_1_to_K_q = seg_masks_q / (self.opt.kl_temp)

        # compute KL divergence between uniform and softmax
        # kl_target is uniform distribution over K classes
        kl_target = torch.ones_like(logits_1_to_K_q) / logits_1_to_K_q.shape[1]
        kld = F.kl_div(
                    input=torch.log_softmax(logits_1_to_K_q, dim=1), 
                    target=kl_target, 
                    reduction="none",
                    ).sum(1)


        # we only backprop loss on uncertain regions
        uncertainty_mask = gamma_masks_q.float()      # 1 where uncertain, 0 where certain

        with torch.no_grad():
            max_softmax_query = torch.softmax(logits_1_to_K_q, dim=1).max(dim=1)[0]     # [bs, h, w]

            print(f"in calculate_kl_uniform_loss, max_softmax_query min mean max: {max_softmax_query.min()}, {max_softmax_query.mean()}, {max_softmax_query.max()}")

            kld_metrics["mean_max_softmax_query_uncertain"] = (max_softmax_query * uncertainty_mask).sum() / uncertainty_mask.sum()
            kld_metrics["mean_max_softmax_query_certain"] = (max_softmax_query * (1-uncertainty_mask)).sum() / (1-uncertainty_mask).sum()

        if not self.opt.no_filtering:
            if uncertainty_mask.sum() == 0:
                loss_kl = torch.zeros_like(kld, requires_grad=True)[0,0,0]
            else:
                loss_kl = (kld * uncertainty_mask).sum() / uncertainty_mask.sum()
        else:
            loss_kl = torch.zeros_like(kld.mean())

        return loss_kl, kld_metrics
    ##########################################################################################################################################################

    ######################################################################################################################################################
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
    ######################################################################################################################################################

    ######################################################################################################################################################
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
    ######################################################################################################################################################

    ######################################################################################################################################################
    def calculate_sup_loss(self, labelled_seg_masks, labelled_seg_masks_proto, labels, labelled_crop_boxes_A):
        sup_metrics = {}

        ### supervised loss ###
        labels = crop_by_box_and_resize(labels.unsqueeze(1).float(), labelled_crop_boxes_A, mode="nearest").squeeze(1).long()
        # NOTE: not divided by self.opt.temperature
        sup_loss = self.sup_xent_loss_fn(labelled_seg_masks, labels)
        sup_loss = sup_loss.mean()

        if labelled_seg_masks_proto is not None:
            sup_loss += self.sup_xent_loss_fn(labelled_seg_masks_proto/self.opt.temperature, labels).mean()
            sup_loss = sup_loss / 2

        with torch.no_grad():
            ### supervised metrics ###
            labelled_segs = torch.argmax(labelled_seg_masks, dim=1)
            # the supervised loss function is calculated over void class as well -> include in metric
            sup_metrics["labelled_miou"] = calculate_miou(labelled_segs, labels, num_classes=self.num_known_classes+1)
            sup_metrics["labelled_accuracy"] = torch.eq(labelled_segs, labels).float().mean()
            if labelled_seg_masks_proto is not None:
                labelled_segs_proto = torch.argmax(labelled_seg_masks_proto, dim=1)
                sup_metrics["labelled_miou_proto"] = calculate_miou(labelled_segs_proto, labels, num_classes=self.num_known_classes+1)
                sup_metrics["labelled_accuracy_proto"] = torch.eq(labelled_segs_proto, labels).float().mean()
        return sup_loss, sup_metrics
    ######################################################################################################################################################

    ######################################################################################################################################################
    def calculate_masking_sup_loss(self, seg_masks, labels, masks):
        masking_metrics = {}

        masking_loss = self.sup_xent_loss_fn(seg_masks, labels)
        # This loss is only calculated over the masked regions
        masking_loss = (masking_loss * masks).sum() / masks.sum()

        with torch.no_grad():
            ### supervised metrics ###
            labelled_segs = torch.argmax(seg_masks, dim=1)
            # the supervised loss function is calculated over void class as well -> include in metric
            masking_metrics["labelled_miou"] = calculate_miou(labelled_segs, labels, num_classes=self.num_known_classes+1)
            masking_metrics["labelled_accuracy"] = torch.eq(labelled_segs, labels).float().mean()

        return masking_loss, masking_metrics
    ######################################################################################################################################################

    @staticmethod
    def semantic_inference(mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    ######################################################################################################################################################
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

        for key in losses:
            if key in ["loss_mask", "loss_dice"]:
                losses[key] = losses[key] * 5
            elif key == "loss_ce":
                losses[key] = losses[key] * 2

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

        return losses, metrics
    ######################################################################################################################################################

    ######################################################################################################################################################
    def calculate_m2f_losses_xent(self, outputs, labels, labelled_crop_boxes_A):
        """
        - outputs["pred_logits"]: [B, Q, K+1]
        - outputs["pred_masks"]: [B, Q, H, W]
        - labels: [B, H, W]
        """
        metrics = {}
        labels = crop_by_box_and_resize(labels.unsqueeze(1).float(), labelled_crop_boxes_A, mode="nearest").squeeze(1).long()

        seg_masks = self.semantic_inference(mask_cls=outputs["pred_logits"], mask_pred=outputs["pred_masks"])

        losses = {}
        losses["loss_m2f_xent"] = self.sup_xent_loss_fn(seg_masks, labels).mean()

        with torch.no_grad():
            segs = torch.argmax(seg_masks, dim=1)
            metrics["m2f_miou"] = calculate_miou(segs, labels, num_classes=self.num_known_classes)
            metrics["m2f_accuracy"] = torch.eq(segs, labels).float().mean()

            metrics["sup_mean_max_softmax"] = torch.softmax(seg_masks, dim=1).max(1)[0].mean().cpu()
            metrics["sup_mean_max_softmax_temp"] = torch.softmax(seg_masks/self.opt.temperature, dim=1).max(1)[0].mean().cpu()

        return losses, metrics
    ######################################################################################################################################################



