import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../")
# import utils
from training.base_trainer import BaseTrainer
from utils.device_utils import to_device
from utils.candr_utils import crop_by_box_and_resize
from utils.gamma_utils import get_gamma_masks
from gammassl_losses import GammaSSLLosses

class Trainer(BaseTrainer):
    """
    Trainer class for GammaSSL.
    """    
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)              # init base trainer class
        
        self.losses = GammaSSLLosses(
                            self.opt, 
                            num_known_classes=len(self.known_class_list),
                            projection_net=self.model.seg_net.projection_net,
                            )

        # for all opts in self.opt, if it starts with "w_" then add it to self.loss_weights
        self.loss_weights = {key: val for key, val in vars(self.opt).items() if key.startswith("w_")}
    
    
    def train_model(self, labelled_dict, unlabelled_dict):
            """
            Trains the model by performing a labelled task and an unlabelled task.
            The unlabelled task implements GammaSSL training.

            Args:
                labelled_dict (dict): A dict containing data from labelled domain.
                unlabelled_dict (dict): A dict containing data from unlabelled domain.

            Returns:
                losses (dict): A dict containing all the losses ready for logging
                metrics (dict): A dict containing all the metrics ready for logging
            """
            losses = {}
            metrics = {}

            self.calculate_prototype_loss_if_needed(losses)

            # calculate losses on labelled domain
            self.perform_labelled_task(labelled_dict, losses, metrics)

            # if performing GammaSSL training, calculate losses on unlabelled domain
            if not self.opt.sup_loss_only:
                self.perform_unlabelled_task(unlabelled_dict, losses, metrics)

            # calculate grads and update model params
            self.update_model(losses)
            return losses, metrics
    
    def calculate_prototype_loss_if_needed(self, losses):
        if self.opt.use_proto_seg:
            # calculate prototypes from random batch
            prototypes = self.model.calculate_batch_prototypes()
            # calculate prototype loss
            losses["loss_p"] = self.losses.calculate_prototype_loss(prototypes)

    def update_model(self, losses):
        # reset model grads
        for network_ids in self.model.optimizers:
            self.model.optimizers[network_ids].zero_grad()

        # calculate grads w.r.t. model params
        model_loss = 0
        for key, loss in losses.items():
            weight = self.loss_weights.get(key.replace("loss_", "w_"), 1)
            model_loss += weight * loss
        (model_loss).backward()
        model_loss = model_loss.item()

        # update model params
        for network_ids in self.model.optimizers:
            self.model.optimizers[network_ids].step()
        for network_ids in self.model.schedulers:
            self.model.schedulers[network_ids].step()

    
    def perform_labelled_task(self, labelled_dict, losses, metrics):
        """
        Calculates losses for batch of data from labelled domain.
        Fills dicts (losses and metrics) with losses and metrics to monitor training.
        
        Args:
            labelled_dict (dict): A dict containing a batch of images, labels and crop locations from labelled domain
            losses (dict): A dict containing previous losses - to be updated
            metrics (dict): A dict containing previous metrics - to be updated
        """
        # data to device
        imgs = to_device(labelled_dict["img"], self.device)
        labels = to_device(labelled_dict["label"], self.device)
        crop_boxes_tA = to_device(labelled_dict["box_A"], self.device)

        # crop and resize images and labels, denoted by transform: tA
        imgs_tA = crop_by_box_and_resize(imgs, crop_boxes_tA)
        labels_tA = crop_by_box_and_resize(labels.unsqueeze(1).float(), crop_boxes_tA, mode="nearest").squeeze(1).long()

        if self.opt.model_arch == "vit_m2f":
            # calculate m2f supervised losses
            m2f_outputs_tA = self.model.seg_net.extract_m2f_output(imgs_tA)
            loss_ce, loss_dice, loss_mask, m2f_metrics = self.losses.calculate_m2f_losses(m2f_outputs_tA, labels_tA)
            losses["loss_ce"], losses["loss_dice"], losses["loss_mask"] = loss_ce, loss_dice, loss_mask
            metrics.update(m2f_metrics)
        else:
            # calculate standard supervised losses
            seg_masks_tA = self.model.get_seg_masks(imgs_tA, high_res=True)
            sup_loss, sup_metrics = self.losses.calculate_sup_loss(seg_masks_tA, labels_tA)
            losses["loss_s"] = sup_loss
            metrics.update(sup_metrics)

    
    def perform_unlabelled_task(self, unlabelled_dict, losses, metrics):
        """
        Calculates losses for batch of images from unlabelled domain.
        Fills dicts (losses and metrics) with losses and metrics to monitor training.

        Args:
            unlabelled_dict (dict): A dict containing a batch of images and crop locations from unlabelled domain
                (imgs_t and imgs_q are the same images, but with different colour-space transformation applied).
            unlabelled_dict (dict): A dict containing a batch of images and crop locations from unlabelled domain.
            losses (dict): A dict containing previous losses - to be updated.
            metrics (dict): A dict containing previous metrics - to be updated.
        
        """
        # data to device
        imgs_t = to_device(unlabelled_dict["img_1"], self.device)
        imgs_q = to_device(unlabelled_dict["img_2"], self.device)
        crop_boxes_tA = to_device(unlabelled_dict["box_A"], self.device)
        crop_boxes_tB = to_device(unlabelled_dict["box_B"], self.device)

        # TARGET branch
        with torch.no_grad():
            # crop and resize images, transform by tA
            imgs_t_tA = crop_by_box_and_resize(imgs_t, crop_boxes_tA)
            # get seg masks from target branch
            seg_masks_t_tA = self.model.get_seg_masks(imgs_t_tA, high_res=True, branch="target")
            # crop and resize seg_masks, transform by tB
            seg_masks_t_tAB = crop_by_box_and_resize(seg_masks_t_tA, crop_boxes_tB)

        # QUERY branch
        # crop and resize images, transform by tB
        imgs_q_tB = crop_by_box_and_resize(imgs_q, crop_boxes_tB)
        # get seg masks from query branch
        if self.opt.use_proto_seg:
            features_q_tB = self.model.seg_net.extract_features(imgs_q_tB)
            seg_masks_q_tB = self.model.proto_segment_features(
                                                    features=features_q_tB, 
                                                    img_spatial_dims=imgs_q.shape[-2:], 
                                                    )
        else:
            seg_masks_q_tB = self.model.get_seg_masks(imgs_q_tB, high_res=True, branch="query")
        # crop and resize seg_masks, transform by tA
        seg_masks_q_tBA = crop_by_box_and_resize(seg_masks_q_tB, crop_boxes_tA)

        # calculate uniformity loss, if needed
        if self.opt.use_proto_seg:
            losses["loss_u"] = self.losses.calculate_uniformity_loss(features_q_tB)

        # update gamma using new seg masks, and calculate binary uncertainty masks for them
        self.model.update_gamma(seg_masks_q=seg_masks_q_tBA, seg_masks_t=seg_masks_t_tAB)
        gamma_masks_q = get_gamma_masks(seg_masks_q_tBA, gamma=self.model.gamma)

        # calculate consistency loss
        losses["loss_c"], ssl_metrics = self.losses.calculate_consistency_loss(seg_masks_t_tAB, seg_masks_q_tBA, gamma_masks_q)
        metrics.update(ssl_metrics)
        