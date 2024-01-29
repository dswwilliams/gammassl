import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import torch.nn as nn
import pickle
import copy
sys.path.append("../")
# import utils
from training.base_trainer import BaseTrainer
from utils.device_utils import to_device
from utils.crop_utils import crop_by_box_and_resize
from gammassl_losses import GammaSSLLosses

class Trainer(BaseTrainer):
    ##########################################################################################################################################################
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)              # init base trainer class
        
        self.losses = GammaSSLLosses(self.opt, self.device, num_known_classes=len(self.known_class_list))

        self.loss_weights = {"w_c": self.opt.w_c, "w_u": self.opt.w_u, "w_p": self.opt.w_p, "w_s": self.opt.w_s, 
                                "w_mask": self.opt.w_mask, "w_dice": self.opt.w_dice, "w_ce": self.opt.w_ce,}

        self.model.gamma = torch.zeros(1, dtype=torch.float32).to(self.device)
    ##########################################################################################################################################################

    
    ##########################################################################################################################################################
    def _train_models(self, labelled_dict, raw_dict):
        """
        steps:
        - model to train
        - if required
            - calculate prototypes
        - perform labelled task
        - if required
            - calculate prototype loss
        - perform unlabelled task
        - backprop, update model and log losses and metrics
        """
        losses = {}
        metrics = {}

        self.model.model_to_train()

        ### calculate prototypes ###
        print("calculating prototypes...")
        if self.opt.use_proto_seg:
            prototypes = self.model.calculate_batch_prototypes()

        ### supervised loss ###
        m2f_losses, sup_metrics = self.perform_labelled_task(labelled_dict)
        for key in sup_metrics:
            metrics[key] = sup_metrics[key].item()
        for key in m2f_losses:                          # loss_mask, loss_dice, loss_ce
            losses[key] = m2f_losses[key]
        
        ### prototype loss ###
        if self.opt.use_proto_seg:
            losses["loss_p"] = self.losses.calculate_prototype_loss(prototypes)

        ### uniformity and consistency losses ###
        losses["loss_u"], losses["loss_c"], ssl_metrics = self.perform_unlabelled_task(raw_dict)
        for key in ssl_metrics:
            metrics[key] = ssl_metrics[key]

        ### zero grad ###
        for network_ids in self.model.optimizers:
            self.model.optimizers[network_ids].zero_grad()

        ### backprop loss ###
        model_loss = 0
        for key, loss in losses.items():
            weight = self.loss_weights.get(key.replace("loss_", "w_"), 1)
            model_loss += weight * loss
        (model_loss).backward()

        ### update weights ###
        for network_ids in self.model.optimizers:
            self.model.optimizers[network_ids].step()
        for network_ids in self.model.schedulers:
            self.model.schedulers[network_ids].step()

        ### logging ###
        model_loss = model_loss.item()
        for key in losses:
            losses[key] = losses[key].item()
        return losses, metrics
    ##########################################################################################################################################################

    ######################################################################################################################################################
    def perform_labelled_task(self, labelled_dict):
        ### (task) labelled task ###
        labelled_imgs = to_device(labelled_dict["img"], self.device)
        labels = to_device(labelled_dict["label"], self.device)
        labelled_crop_boxes_A = to_device(labelled_dict["box_A"], self.device)

        if self.opt.model_arch == "vit_m2f":
            m2f_outputs = self.segment_labelled_imgs(labelled_imgs, labelled_crop_boxes_A)
            ### (losses) compute supervised losses and their gradients ###
            losses, metrics = self.losses.calculate_m2f_losses(m2f_outputs, labels, labelled_crop_boxes_A)
        elif self.opt.model_arch == "deeplab":
            labelled_imgs_A = crop_by_box_and_resize(labelled_imgs, labelled_crop_boxes_A)  
            labelled_seg_masks_A = self.model.seg_net.get_seg_masks(labelled_imgs_A, include_void=False, high_res=True)
            sup_loss, metrics = self.losses.calculate_sup_loss(labelled_seg_masks_A, labels, labelled_crop_boxes_A)
            losses = {"loss_s": sup_loss}

        return losses, metrics
    ######################################################################################################################################################
    

    ######################################################################################################################################################
    def perform_unlabelled_task(self, raw_dict):
        raw_imgs_t = to_device(raw_dict["img_1"], self.device)
        raw_imgs_q = to_device(raw_dict["img_2"], self.device)
        raw_crop_boxes_A = to_device(raw_dict["box_A"], self.device)
        raw_crop_boxes_B = to_device(raw_dict["box_B"], self.device)

        ### target branch ###
        with torch.no_grad():
            raw_imgs_t_tA = crop_by_box_and_resize(raw_imgs_t, raw_crop_boxes_A)
            if self.opt.frozen_target:
                # TODO: can we bury whether the target is frozen or not in the model?
                seg_masks_t_tA = self.model.target_seg_net.get_target_seg_masks(raw_imgs_t_tA, include_void=False, high_res=True)
            else:
                seg_masks_t_tA = self.model.seg_net.get_target_seg_masks(raw_imgs_t_tA, include_void=False, high_res=True)
            seg_masks_t_tAB = crop_by_box_and_resize(seg_masks_t_tA, raw_crop_boxes_B)

        ### query branch ###
        raw_imgs_q_tB = crop_by_box_and_resize(raw_imgs_q, raw_crop_boxes_B)
        if self.opt.use_proto_seg:
            raw_features_q_tB = self.model.seg_net.extract_features(raw_imgs_q_tB)
            seg_masks_q_tB = self.model.proto_segment_features(
                                                    features=raw_features_q_tB, 
                                                    img_spatial_dims=raw_imgs_q.shape[-2:], 
                                                    )
        else:
            seg_masks_q_tB = self.model.seg_net.get_query_seg_masks(raw_imgs_q_tB, include_void=False, high_res=True)
        seg_masks_q_tBA = crop_by_box_and_resize(seg_masks_q_tB, raw_crop_boxes_A)

        ### uniformity loss ###
        if self.opt.use_proto_seg:
            loss_u = self.losses.calculate_uniformity_loss(features=raw_features_q_tB, projection_net=self.model.seg_net.projection_net)
        else:
            loss_u = torch.zeros(1).to(self.device).squeeze()

        ### calculating gamma ###
        if not self.opt.no_filtering:
            self.model.update_gamma(seg_masks_q=seg_masks_q_tBA, seg_masks_t=seg_masks_t_tAB)

        ### get gamma masks ###
        gamma_masks_q = self.evaluate_seg_masks_with_gamma(seg_masks_q_tBA)

        ### consistency loss ###
        loss_c, ssl_metrics = self.losses.calculate_ssl_loss(seg_masks_t_tAB, seg_masks_q_tBA, gamma_masks_q)


        return loss_u, loss_c, ssl_metrics
    ######################################################################################################################################################

    ######################################################################################################################################################
    def segment_labelled_imgs(self, labelled_imgs, labelled_crop_boxes_A):
        ### perform labelled task ### 
        labelled_imgs_A = crop_by_box_and_resize(labelled_imgs.detach(), labelled_crop_boxes_A)  

        """ learned encoder shared with raw task, pretrained segmentation head """
        output = self.model.seg_net.extract_m2f_output(labelled_imgs_A)

        return output
    ######################################################################################################################################################


    ######################################################################################################################################################
    def evaluate_seg_masks_with_gamma(self, seg_masks_q_tBA):
        # NOTE: we dont want to update gamma when this is called -> use gamma.detach()
        gamma_seg_masks_q_tBA = self.segmasks2gammasegmasks(seg_masks_q_tBA, gamma=self.model.gamma.detach())
        gamma_masks_q_tBA = torch.eq(torch.argmax(gamma_seg_masks_q_tBA, dim=1), len(self.known_class_list)).float()
        return gamma_masks_q_tBA
    ######################################################################################################################################################


    ######################################################################################################################################################
    def segmasks2gammasegmasks(self, seg_masks, gamma):
        bs, _, h, w = seg_masks.shape
        device = seg_masks.device
        gammas = gamma * torch.ones(bs, 1, h, w).to(device)      # shape: [bs, 1, h, w]
        gamma_seg_masks = torch.cat((seg_masks, gammas), dim=1)             # shape: [bs, K+1, h, w]
        return gamma_seg_masks
    ######################################################################################################################################################

