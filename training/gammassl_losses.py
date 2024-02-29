import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
from utils.training_metric_utils import get_consistency_metrics, calculate_supervised_metrics
import sys
sys.path.append("../")
from models.mask2former_loss import SetCriterion
from models.matcher import HungarianMatcher
from utils.m2f_utils import semantic_inference


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
        
class UniformityLoss(nn.Module):
    def __init__(self, kernel_size, stride, projection_net, rbf_t=2):
        """
        Calculates the uniformity loss, which maximises the uniformity of the given features.
        Based on https://arxiv.org/pdf/2005.10242.pdf.
        
        Args:
            kernel_size (int): Size of the kernel used to downsample the features.
            stride (int): Stride of the kernel used to downsample the features.
            rbf_t (float): Temperature for the radial basis function in the uniformity loss fn.
            projection_net (nn.Module): A network that projects the features.
        """
        super(UniformityLoss, self).__init__()


        self.kernel_size = kernel_size
        self.stride = stride
        self.rbf_t = rbf_t
        if projection_net is not None:
            self.projection_net = projection_net
        else:
            self.projection_net = nn.Identity()

    def forward(self, features):
        # downsample
        low_res_features = F.avg_pool2d(
                            features,
                            kernel_size=(self.kernel_size, self.kernel_size),
                            stride=self.stride,
                            )

        # project (if required)
        low_res_features = self.projection_net(low_res_features)

        # normalise
        bs, feature_dim, h, w = low_res_features.shape
        low_res_features = F.normalize(low_res_features.permute(0,2,3,1).reshape(bs*h*w, feature_dim), dim=1, p=2)

        # calculate uniformity loss
        loss_u = torch.pdist(low_res_features, p=2).pow(2).mul(-self.rbf_t).exp().mean().log()
        return loss_u
    

class GammaSSLLosses:
    """
    Implements loss functions for the different types of GammaSSL training.
    """
    def __init__(self, opt, num_known_classes, projection_net):
        self.opt = opt
        self.num_known_classes = num_known_classes

        self.temperature = opt.temperature

        # determine whether to include void class in loss
        ignore_index = -100 if self.opt.include_void else self.num_known_classes

        self.sharpen = Sharpener(temperature=opt.sharpen_temp)
        self.hard_xent = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.soft_xent = SoftProbCrossEntropy(dim=1, reduction="none")
        self.uniformity_loss_fn = UniformityLoss(
                                    kernel_size=opt.uniformity_kernel_size,
                                    stride=opt.uniformity_stride,
                                    projection_net=projection_net,
                                    )

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
            seg_masks_t (torch.Tensor): Segmentation masks for target branch [bs, K, H, W]
            seg_masks_q (torch.Tensor): Segmentation masks for query branch [bs, K, H, W]
            gamma_masks_q (torch.Tensor): Masks defining pixels where seg_masks_q is uncertain [bs, H, W]
            input_masks_q (torch.Tensor): Masks applied to input images [bs, H, W]

        Returns:
            loss_c (torch.Tensor): consistency loss [1]
            consistency_metrics (dict): dictionary of training metrics
        """
        _sharpen = self.sharpen if self.opt.sharpen_temp is not None else lambda x: x


        p_y_given_x_t = _sharpen(torch.softmax(seg_masks_t / self.temperature, dim=1))
        p_y_given_x_q = torch.softmax(seg_masks_q / self.temperature, dim=1)
        xent = self.soft_xent(p_y_given_x_t, p_y_given_x_q)

        
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

        consistency_metrics = get_consistency_metrics(
                                            p_y_given_x_t=p_y_given_x_t, 
                                            p_y_given_x_q=p_y_given_x_q,
                                            certainty_masks=certainty_mask,
                                            detailed_metrics=self.opt.detailed_metrics,
                                            )

        return loss_c, consistency_metrics
    
    def calculate_uniformity_loss(self, features):
        return self.uniformity_loss_fn(features)


    @staticmethod
    def calculate_prototype_loss(prototypes):
        """
        Calculates the prototype loss, which minimises the maximum cosine similarity between prototypes.

        Args:
            prototypes (torch.Tensor): Normalised prototypes [K, D]
        Returns:
            loss (torch.Tensor): prototype loss [1]
        """

        # get cosine similarity matrix in range [0, 2]
        sim_matrix = torch.matmul(prototypes, prototypes.t()) + 1

        # Remove diagonal (i.e. self-similarity)
        sim_matrix.fill_diagonal_(-1)

        # for each prototype, get the maximum inter-prototype similarity
        loss = sim_matrix.max(dim=1).values

        return loss.mean()
    

    def calculate_sup_loss(self, labelled_seg_masks, labels):
        """
        Calculates the supervised loss as the cross-entropy with a hard target (i.e. labels).

        Args:
            labelled_seg_masks (torch.Tensor): Segmentation masks for labelled images [bs, K, H, W]
            labels (torch.Tensor): Labels for labelled images [bs, H, W]
        Returns:
            sup_loss (torch.Tensor): supervised loss [1]
            sup_metrics (dict): dictionary of training metrics, (namely labelled_miou and labelled_accuracy)
        """
        metrics = {}

        sup_loss = self.hard_xent(labelled_seg_masks, labels)
        sup_loss = sup_loss.mean()

        # calculating training metrics
        with torch.no_grad():
            metrics = calculate_supervised_metrics(labelled_seg_masks, labels)
        return sup_loss, metrics

    
    def calculate_m2f_losses(self, m2f_outputs, labels):
        """
        Calculates the supervised Mask2Former losses.

        Args:
        - m2f_outputs (dict) containing:
                - pred_logits: [bs, Q, K+1]
                - pred_masks: [bs, Q, H, W]
        - labels: [bs, H, W]
        Returns:
        - loss_ce (torch.Tensor): supervised cross-entropy loss [1]
        - loss_dice (torch.Tensor): supervised dice loss [1]
        - loss_mask (torch.Tensor): supervised mask loss [1]
        - metrics (dict): dictionary of training metrics
        """

        metrics = {}

        bs = labels.shape[0]
        K = self.num_known_classes

        # creating targets for m2f losses
        label_masks = F.one_hot(labels, num_classes=K+1).permute(0, 3, 1, 2)[:,:K,:,:].long()
        class_labels = torch.arange(K).unsqueeze(0).expand(bs, -1)
        targets = []    
        for batch_no in range(bs):
            target = {}
            target["labels"] = class_labels[batch_no]             # [K]
            target["masks"] = label_masks[batch_no]               # [K, H, W]
            targets.append(target)

        # calculate losses
        losses = self.m2f_criterion(m2f_outputs, targets)
        
        # separate out and weight losses
        loss_ce = losses["loss_ce"] * 2 if "loss_ce" in losses else None
        loss_dice = losses["loss_dice"] * 5 if "loss_dice" in losses else None
        loss_mask = losses["loss_mask"] * 5 if "loss_mask" in losses else None
        del losses

        # calculate metrics
        with torch.no_grad():
            seg_masks = semantic_inference(mask_cls=m2f_outputs["pred_logits"], mask_pred=m2f_outputs["pred_masks"])
            metrics = calculate_supervised_metrics(seg_masks, labels)
        return loss_ce, loss_dice, loss_mask, metrics