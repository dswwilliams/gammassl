import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
import numpy as np
sys.path.append("../")
# import utils
from training.base_trainer import BaseTrainer
from utils.device_utils import to_device
from utils.crop_utils import crop_by_box_and_resize
from utils.gamma_utils import get_gamma_masks
from gammassl_losses import GammaSSLLosses

class Trainer(BaseTrainer):
    
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)              # init base trainer class
        
        self.losses = GammaSSLLosses(self.opt, self.device, num_known_classes=len(self.known_class_list))

        # for all opts in self.opt, if it starts with "w_" then add it to self.loss_weights
        self.loss_weights = {key: val for key, val in vars(self.opt).items() if key.startswith("w_")}
    
    
    def train_model(self, labelled_dict, raw_dict):
        losses = {}
        metrics = {}


        self.calculate_prototype_loss_if_needed(losses)

        self.perform_labelled_task(labelled_dict, losses, metrics)

        if not self.opt.sup_loss_only:
            self.perform_unlabelled_task(raw_dict, losses, metrics)

        self.update_model(losses)
        return losses, metrics 
    
    def calculate_prototype_loss_if_needed(self, losses):
        if self.opt.use_proto_seg:
            prototypes = self.model.calculate_batch_prototypes()
            losses["loss_p"] = self.losses.calculate_prototype_loss(prototypes)

    def update_model(self, losses):
        ### zero grad ###
        for network_ids in self.model.optimizers:
            self.model.optimizers[network_ids].zero_grad()

        ### backprop loss ###
        model_loss = 0
        for key, loss in losses.items():
            weight = self.loss_weights.get(key.replace("loss_", "w_"), 1)
            model_loss += weight * loss
        (model_loss).backward()
        model_loss = model_loss.item()

        ### update weights ###
        for network_ids in self.model.optimizers:
            self.model.optimizers[network_ids].step()
        for network_ids in self.model.schedulers:
            self.model.schedulers[network_ids].step()

    
    def perform_labelled_task(self, labelled_dict, losses, metrics):
        ### (task) labelled task ###
        labelled_imgs = to_device(labelled_dict["img"], self.device)
        labels = to_device(labelled_dict["label"], self.device)
        labelled_crop_boxes_A = to_device(labelled_dict["box_A"], self.device)

        labelled_imgs_A = crop_by_box_and_resize(labelled_imgs.detach(), labelled_crop_boxes_A)

        if self.opt.model_arch == "vit_m2f":
            m2f_outputs = self.model.seg_net.extract_m2f_output(labelled_imgs_A)
            loss_ce, loss_dice, loss_mask, m2f_metrics = self.losses.calculate_m2f_losses(
                                                                                m2f_outputs, 
                                                                                labels, 
                                                                                labelled_crop_boxes_A
                                                                                )
            losses["loss_ce"], losses["loss_dice"], losses["loss_mask"] = loss_ce, loss_dice, loss_mask
            metrics.update(m2f_metrics)

        elif self.opt.model_arch == "deeplab":
            labelled_imgs_A = crop_by_box_and_resize(labelled_imgs, labelled_crop_boxes_A)  
            labelled_seg_masks_A = self.model.get_seg_masks(labelled_imgs_A, high_res=True)
            sup_loss, sup_metrics = self.losses.calculate_sup_loss(labelled_seg_masks_A, labels, labelled_crop_boxes_A)
            losses["loss_s"] = sup_loss
            metrics.update(sup_metrics)

    
    def perform_unlabelled_task(self, raw_dict, losses, metrics):
        raw_imgs_t = to_device(raw_dict["img_1"], self.device)
        raw_imgs_q = to_device(raw_dict["img_2"], self.device)
        raw_crop_boxes_A = to_device(raw_dict["box_A"], self.device)
        raw_crop_boxes_B = to_device(raw_dict["box_B"], self.device)

        # get seg_masks from TARGET branch
        with torch.no_grad():
            raw_imgs_t_tA = crop_by_box_and_resize(raw_imgs_t, raw_crop_boxes_A)
            seg_masks_t_tA = self.model.get_seg_masks(raw_imgs_t_tA, high_res=True, branch="target")
            seg_masks_t_tAB = crop_by_box_and_resize(seg_masks_t_tA, raw_crop_boxes_B)

        # get seg_masks from QUERY branch
        raw_imgs_q_tB = crop_by_box_and_resize(raw_imgs_q, raw_crop_boxes_B)
        if self.opt.use_proto_seg:
            raw_features_q_tB = self.model.seg_net.extract_features(raw_imgs_q_tB)
            seg_masks_q_tB = self.model.proto_segment_features(
                                                    features=raw_features_q_tB, 
                                                    img_spatial_dims=raw_imgs_q.shape[-2:], 
                                                    )
        else:
            seg_masks_q_tB = self.model.get_seg_masks(raw_imgs_q_tB, high_res=True, branch="query")
        seg_masks_q_tBA = crop_by_box_and_resize(seg_masks_q_tB, raw_crop_boxes_A)

        ### uniformity loss ###
        if self.opt.use_proto_seg:
            losses["loss_u"] = self.losses.calculate_uniformity_loss(
                                                        raw_features_q_tB,
                                                        self.model.seg_net.projection_net
                                                        )

        ### calculating gamma ###
        if not self.opt.no_filtering:
            self.model.update_gamma(seg_masks_q=seg_masks_q_tBA, seg_masks_t=seg_masks_t_tAB)

        ### get gamma masks ###
        gamma_masks_q = get_gamma_masks(seg_masks_q_tBA, gamma=self.model.gamma)

        ### consistency loss ###
        losses["loss_c"], ssl_metrics = self.losses.calculate_ssl_loss(seg_masks_t_tAB, seg_masks_q_tBA, gamma_masks_q)
        metrics.update(ssl_metrics)