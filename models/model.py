import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device_utils import to_device
from utils.crop_utils import crop_by_box_and_resize
from utils.downsampling_utils import ClassWeightedModalDownSampler
from tqdm import tqdm
import numpy as np
import pickle
from ema_pytorch import EMA
import copy
import sys
sys.path.append("../")


class SegmentationModel(nn.Module):
    """ class that deals with defining and updating: neural networks, prototypes and gamma """
    ##########################################################################################################################################################
    def __init__(self, device, opt, known_class_list, crop_size):
        super().__init__()
        self.opt = opt
        self.device = device
        self.num_known_classes = len(known_class_list)
        self.crop_size = crop_size

        if self.opt.encoder == "dino_repo":
            from models.vit_dino_repo_seg_net import ViTDINOSegNet as SegNet
        elif self.opt.encoder == "dino_repo_m2f":
            from models.vit_dino_repo_m2f_seg_net import ViTDINOSegNet as SegNet

        

        self.seg_net = SegNet(device, opt, num_known_classes=self.num_known_classes)
        if self.opt.lora_rank is not None:
            import loralib as lora
            lora.mark_only_lora_as_trainable(self.seg_net.backbone)


        if self.opt.frozen_target:
            self.target_seg_net = copy.deepcopy(self.seg_net)
            self.target_seg_net.to(self.device)
            # load in network trained with sup task only
            checkpoint = torch.load(self.opt.frozen_target_save_path, map_location=self.device)
            self.target_seg_net.backbone.load_state_dict(checkpoint["backbone"], strict=False)
            self.target_seg_net.decode_head.load_state_dict(checkpoint["decode_head"])
            self.target_seg_net.eval()      # should stay in eval mode
        elif self.opt.use_ema_target_net:
            self.ema_seg_net = EMA(
                                self.seg_net,
                                beta = self.opt.ema_beta,                          # exponential moving average factor
                                update_after_step = 0,                             # only after this number of .update() calls will it start updating
                                update_every = self.opt.ema_update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
                                inv_gamma=1.0,
                                power=1,                                           # gamma < 1 means model gets further from online model through time (beta gets closer to 1)
                                )


        ### define prototypes ###
        if self.opt.prototypes_path is not None:
            print("loading prototypes from ->", self.opt.prototypes_path)
            if self.opt.prototypes_path[-4:] == ".pkl":
                from utils.prototype_utils import load_prototypes_from_pkl
                self.dataset_prototypes = load_prototypes_from_pkl(self.opt.prototypes_path, self.device)
            else:
                checkpoint = torch.load(self.opt.prototypes_path, map_location=self.device)
                self.dataset_prototypes = checkpoint["prototypes"]

        else:
            self.dataset_prototypes = None
        
        self.batch_prototypes = None
        self.old_prototypes = None
        self.gamma = None

        self.class_weighted_modal_downsampling = ClassWeightedModalDownSampler(known_class_list=known_class_list)

        from utils.hypersphere_prototype_utils import Extract_HyperSpherePrototypes, Segment_via_HyperSpherePrototypes
        self.extract_prototypes = Extract_HyperSpherePrototypes(num_known_classes=self.num_known_classes)
        self.segment_via_prototypes = Segment_via_HyperSpherePrototypes()

        from datasets.cityscapes_bdd_dataset import CityscapesxBDDDataset
        _proto_dataset = CityscapesxBDDDataset

        # for training
        train_proto_dataset = _proto_dataset(
                                    self.opt.cityscapes_dataroot, 
                                    self.opt.unlabelled_dataroot, 
                                    self.opt.no_transforms,
                                    self.opt.min_crop_ratio,
                                    self.opt.max_crop_ratio,
                                    add_resize_noise=self.opt.use_resize_noise,
                                    only_labelled=True,
                                    use_imagenet_norm=self.opt.use_imagenet_norm,
                                    no_colour=self.opt.no_colour,
                                    crop_size=self.crop_size,
                                    )
        self.train_proto_dataloader = torch.utils.data.DataLoader(dataset=train_proto_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, drop_last=True)
        self.train_proto_iterator = iter(self.train_proto_dataloader)

        # for validation
        self.val_proto_dataset = _proto_dataset(
                                        labelled_dataroot=self.opt.cityscapes_dataroot, 
                                        raw_dataroot=self.opt.unlabelled_dataroot, 
                                        no_appearance_transform=True,
                                        min_crop_ratio=self.opt.min_crop_ratio,
                                        max_crop_ratio=self.opt.max_crop_ratio,
                                        add_resize_noise=False,
                                        only_labelled=True,
                                        use_imagenet_norm=self.opt.use_imagenet_norm,
                                        no_colour=True,
                                        crop_size=self.crop_size,
                                        )

        
        ### init optimizers ###
        self._init_optimizers()

        # end of init
        ##########################################################################################################################################################


    def _init_optimizers(self):
        _optimizer = torch.optim.AdamW

        # setup optmisers for each model part 
        self.optimizers = {}

        if self.opt.lr_backbone is not None:
            backbone_lr = self.opt.lr_backbone
        else:
            backbone_lr = self.opt.lr

        if self.opt.use_vit_adapter:
            # not updating ViT blocks, pos_embed_interpolated, patch_embed, cls_token
            params_to_update = []
            names_of_params_not_to_update = []
            for name, param in self.seg_net.backbone.named_parameters():
                if "blocks" in name:
                    names_of_params_not_to_update.append(name)
                elif name == "pos_embed_interpolated":
                    names_of_params_not_to_update.append(name)
                elif "patch_embed" in name:
                    names_of_params_not_to_update.append(name)
                elif name == "cls_token":
                    names_of_params_not_to_update.append(name)
                else:
                    params_to_update.append(param)

            self.optimizers["backbone"] = _optimizer(
                                            params=params_to_update, 
                                            lr=backbone_lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),         # default
                                            )
        elif self.opt.lora_rank is not None:
            backbone_trainable_params = []
            for name, param in self.seg_net.backbone.named_parameters():
                if "lora" in name:
                    backbone_trainable_params.append(param)


            self.optimizers["backbone"] = _optimizer(
                                            params=backbone_trainable_params, 
                                            lr=backbone_lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),         # default
                                            )
        else:
            self.optimizers["backbone"] = _optimizer(
                                            params=list(self.seg_net.backbone.parameters()), 
                                            lr=backbone_lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),         # default
                                            )
        if self.seg_net.neck is not None:
            self.optimizers["neck"] = _optimizer(
                                            params=list(self.seg_net.neck.parameters()), 
                                            lr=self.opt.lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),         # default
                                            )
        self.optimizers["decode_head"] = _optimizer(
                                            params=list(self.seg_net.decode_head.parameters()), 
                                            lr=self.opt.lr*self.opt.lr_mult, 
                                            weight_decay=self.opt.model_weight_decay*self.opt.decay_mult,
                                            betas=(0.9, 0.999),         # default
                                            )
        if self.seg_net.seg_head is not None:
            self.optimizers["seg_head"] = _optimizer(
                                                params=list(self.seg_net.seg_head.parameters()), 
                                                lr=self.opt.lr*self.opt.lr_mult, 
                                                weight_decay=self.opt.model_weight_decay*self.opt.decay_mult,
                                                betas=(0.9, 0.999),         # default
                                                )
        if self.seg_net.projection_net is not None:
            self.optimizers["projection_net"] = _optimizer(
                                        params=list(self.seg_net.projection_net.parameters()), 
                                        lr=self.opt.lr*self.opt.lr_mult, 
                                        weight_decay=self.opt.model_weight_decay*self.opt.decay_mult,
                                        betas=(0.9, 0.999),         # default
                                        )

        # setup learning rate scheduling
        self.schedulers = {}
        if self.opt.lr_policy == None:
            pass
        if self.opt.lr_policy == "poly":
            for network in self.optimizers:
                self.schedulers[network] = torch.optim.lr_scheduler.PolynomialLR(self.optimizers[network], power=1.0, total_iters=self.opt.total_iters)

        elif self.opt.lr_policy == "warmup_poly":
            if (self.opt.warmup_ratio is not None) and (self.opt.n_warmup_iters) is not None:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizers[network], start_factor=self.opt.warmup_ratio, end_factor=1, total_iters=self.opt.n_warmup_iters)
                decay_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizers[network], power=1.0, total_iters=self.opt.total_iters)
                self.schedulers[network] = torch.optim.lr_scheduler.SequentialLR(self.optimizers[network], [warmup_scheduler, decay_scheduler], milestones=[self.opt.n_warmup_iters])
            else:
                raise ValueError("Warmup policy selected but warmup parameters not specified")

    def model_to_train(self):
        self.seg_net.train()

    def model_to_eval(self):
        self.seg_net.eval()

    def proto_segment_features(self, features, img_spatial_dims=None, use_dataset_prototypes=False, skip_projection=False, include_void=False):
        if use_dataset_prototypes:
            prototypes = self.dataset_prototypes
        else:
            prototypes = self.batch_prototypes
        
        if skip_projection or self.opt.skip_projection:
            proj_features = features
        else:
            proj_features = self.seg_net.projection_net(features)

        if include_void:
            gamma_input = self.gamma
        else:
            gamma_input = None

        seg_masks_q_tB, mean_sim_to_NNprototype_q = self.segment_via_prototypes(
                                                                            proj_features,
                                                                            prototypes.detach(),         # NB: to prevent backprop through them for this task
                                                                            gamma=gamma_input,
                                                                            output_metrics=True,
                                                                            )
        if img_spatial_dims is not None:
            H,W = img_spatial_dims
            seg_masks_q_tB = F.interpolate(seg_masks_q_tB, size=(H,W), mode="bilinear", align_corners=True)
        return seg_masks_q_tB, mean_sim_to_NNprototype_q

    def proto_segment_imgs(self, imgs, use_dataset_prototypes=False, output_spread=False, include_void=False, masks=None, skip_projection=False):
        if skip_projection or self.opt.skip_projection:
            features = self.seg_net.extract_features(imgs, masks=masks)
        else:
            features = self.seg_net.extract_proj_features(imgs, masks=masks)

        if use_dataset_prototypes:
            prototypes = self.dataset_prototypes
        else:
            prototypes = self.batch_prototypes
        H,W = imgs.shape[2:]

        if include_void:
            gamma_input = self.gamma
        else:
            gamma_input = None
        # NB: detach prototypes to prevent backprop through them for this task
        seg_masks_q_tB, mean_sim_to_NNprototype_q = self.segment_via_prototypes(
                                                                            features, 
                                                                            prototypes.detach(), 
                                                                            gamma=gamma_input,
                                                                            output_metrics=output_spread,
                                                                            )
        seg_masks_q_tB = F.interpolate(seg_masks_q_tB, size=(H,W), mode="bilinear", align_corners=True)
        if output_spread:
            return seg_masks_q_tB, mean_sim_to_NNprototype_q
        else:
            return seg_masks_q_tB
    

    def calculate_batch_prototypes(self):
        """ Calculate prototypes from batch of labelled images """

        ### reading in data from dedicated dataloader for prototypes ###
        try:
            labelled_dict,_ = next(self.train_proto_iterator)
            labelled_imgs = to_device(labelled_dict["img"], self.device)
            labels = to_device(labelled_dict["label"], self.device)
            labelled_crop_boxes_A = to_device(labelled_dict["box_A"], self.device)
        except:
            self.train_proto_iterator = iter(self.train_proto_dataloader)
            labelled_dict,_ = next(self.train_proto_iterator)
            labelled_imgs = to_device(labelled_dict["img"], self.device)
            labels = to_device(labelled_dict["label"], self.device)
            labelled_crop_boxes_A = to_device(labelled_dict["box_A"], self.device)
        
        # getting data in right form
        labelled_imgs_A = crop_by_box_and_resize(labelled_imgs, labelled_crop_boxes_A)
        labels_A = crop_by_box_and_resize(labels.unsqueeze(1).float(), labelled_crop_boxes_A, mode="nearest").squeeze(1).long()
        

        # calculate prototypes from images and labels
        if not self.opt.skip_projection:
            labelled_features_A = self.seg_net.extract_proj_features(labelled_imgs_A)
        else:
            labelled_features_A = self.seg_net.extract_features(labelled_imgs_A)


        # upsample labels to multiple of feature dimension in order to use class_weighted_modal_downsampling (it needs an integer downsmple factor)
        feature_spatial_dim = labelled_features_A.shape[-1]
        labels_spatial_dim = labels_A.shape[-1]
        new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
        labels_A = F.interpolate(labels_A.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
        low_res_labels_A = self.class_weighted_modal_downsampling(labels_A, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)

        prototypes = self.extract_prototypes(labelled_features_A, low_res_labels_A, output_metrics=False)

        # if class doesnt exist in batch, use previous prototype
        for k in range(prototypes.shape[1]):
            proto_for_class = prototypes[:,k]
            # if all values in prototype are zero, there were no features of that class, therefore use previous prototype from that class
            if torch.eq(proto_for_class, 0).all() and self.old_prototypes is not None:
                prototypes[:,k] = self.old_prototypes[:,k].to(prototypes.device)
                # mean_sim_w_GTprototypes[k] = self.old_mean_sim_w_GTprototypes[k].to(prototypes.device)

        # assign new batch prototypes to class variable
        self.batch_prototypes = prototypes

        self.old_prototypes = prototypes.detach().cpu()
        # self.old_mean_sim_w_GTprototypes = mean_sim_w_GTprototypes.detach().cpu()

        return self.batch_prototypes
    
    @torch.no_grad()
    def calculate_dataset_prototypes(self):
        if self.opt.prototypes_path is None:
            """ Calculate prototype from entire dataset """
            dataloader = torch.utils.data.DataLoader(self.val_proto_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, drop_last=True)

            self.device = next(self.seg_net.parameters()).device
            
            iterator = tqdm(dataloader)
            print("calculating dataset prototypes...")
            prototypes_sum = 0
            for step, (labelled_dict,_) in enumerate(iterator):
                labelled_imgs = labelled_dict["img"].to(self.device)
                labels = labelled_dict["label"].to(self.device) 

                if self.opt.use_deep_features:
                    # upsample labels to multiple of feature dimension in order to use class_weighted_modal_downsampling (it needs an integer downsmple factor)
                    feature_spatial_dim = 64
                    labels_spatial_dim = labels.shape[-1]
                    new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
                    labels = F.interpolate(labels.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
                    low_res_labels = self.class_weighted_modal_downsampling(labels, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)
                else:
                    # NOTE this is SOOO small (even smaller than patch embeddings)
                    feature_spatial_dim = 8
                    labels_spatial_dim = labels.shape[-1]
                    new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
                    labels = F.interpolate(labels.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
                    low_res_labels = self.class_weighted_modal_downsampling(labels, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)

                if not self.opt.skip_projection:
                    labelled_features = self.seg_net.extract_proj_features(labelled_imgs)
                else:
                    labelled_features = self.seg_net.extract_features(labelled_imgs)
                prototypes = self.extract_prototypes(labelled_features, low_res_labels, output_metrics=False)

                prototypes_sum += prototypes

                if step > 500:
                    break

            prototypes = F.normalize(prototypes_sum, dim=0, p=2)          # shape: [feature_length, num_known_classes]

            self.dataset_prototypes = prototypes
            return self.dataset_prototypes
        else:
            return None
        
    @torch.no_grad()
    def calculate_dataset_prototypes_target(self):
        if self.opt.prototypes_path is None:
            """ Calculate prototype from entire dataset """
            dataloader = torch.utils.data.DataLoader(self.val_proto_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, drop_last=True)

            self.device = next(self.seg_net.parameters()).device
            
            iterator = tqdm(dataloader)
            print("calculating dataset prototypes...")
            prototypes_sum = 0
            for step, (labelled_dict,_) in enumerate(iterator):
                labelled_imgs = labelled_dict["img"].to(self.device)
                labels = labelled_dict["label"].to(self.device) 

                if self.opt.use_deep_features:
                    # upsample labels to multiple of feature dimension in order to use class_weighted_modal_downsampling (it needs an integer downsmple factor)
                    feature_spatial_dim = 64
                    labels_spatial_dim = labels.shape[-1]
                    new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
                    labels = F.interpolate(labels.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
                    low_res_labels = self.class_weighted_modal_downsampling(labels, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)
                else:
                    # NOTE this is SOOO small (even smaller than patch embeddings)
                    feature_spatial_dim = 16
                    labels_spatial_dim = labels.shape[-1]
                    new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
                    labels = F.interpolate(labels.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
                    low_res_labels = self.class_weighted_modal_downsampling(labels, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)

                # if not self.opt.skip_projection:
                #     labelled_features = self.seg_net.extract_proj_features(labelled_imgs)
                # else:
                #     labelled_features = self.seg_net.extract_features(labelled_imgs)
                labelled_imgs = labelled_imgs.to("mps")
                self.target_seg_net = self.target_seg_net.to("mps")
                labelled_features = self.target_seg_net.extract_features(labelled_imgs, use_deep_features=False)
                labelled_features = labelled_features.to("cpu")
                print(f"Labelled features shape: {labelled_features.shape}")
                print(f"low_res_labels shape: {low_res_labels.shape}")
                prototypes = self.extract_prototypes(labelled_features, low_res_labels, output_metrics=False)

                prototypes_sum += prototypes

                if step > 500:
                    break

            prototypes = F.normalize(prototypes_sum, dim=0, p=2)          # shape: [feature_length, num_known_classes]

            self.dataset_prototypes = prototypes
            return self.dataset_prototypes
        else:
            return None
    
    ##########################################################################################################################################################
    def update_gamma(self, seg_masks_q, seg_masks_t, raw_masks_q=None, raw_masks_t=None):

        segs_q, segs_t = seg_masks_q.detach().argmax(1), seg_masks_t.detach().argmax(1)
        consistency_masks = torch.eq(segs_q, segs_t).float()
        # print(f"in update gamma, consistency_masks mean: {consistency_masks.mean()}")
        # i.e. proxy for mean accuracy
        # if self.opt.loss_c_unmasked_only:
        #     # update gamma based on mean consistency of unmasked pixels only
        #     H, W = consistency_masks.shape[-2:] 
        #     raw_masks_q = raw_masks_q.reshape(-1, H//14, W//14)
        #     raw_masks_q = F.interpolate(raw_masks_q.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1).long()
        #     # ((consistency_masks * (1-raw_masks_q.float())).sum()/ (1-raw_masks_q.float()).sum()).cpu()
        #     raw_masks_q = raw_masks_q.to(consistency_masks.device)
        #     mean_consistency = (consistency_masks * (1-raw_masks_q.float())).sum() / (1-raw_masks_q.float()).sum()
        # else:
        mean_consistency = consistency_masks.mean()

        with torch.no_grad():
            ### calculate gamma so that p(certain) = p(consistent) ###
            reject_proportion = (1 - mean_consistency)   # reject (1 - estimated accuracy)
            # NOTE: think about what scores we are giving it
            if (self.opt.gamma_scaling is None) or (self.opt.gamma_scaling == "None"):
                sims_per_pixel = torch.max(seg_masks_q.detach(), dim=1)[0]         # shape [bs, h, w]
                print(f"in update_gamma: ms_imgs_q min mean max: {sims_per_pixel.min()}, {sims_per_pixel.mean()}, {sims_per_pixel.max()}")
            elif self.opt.gamma_scaling == "softmax":
                # use temperature weighted softmax scores (like in testing)
                sims_per_pixel = torch.max(torch.softmax(seg_masks_q.detach()/self.opt.gamma_temp, dim=1), dim=1)[0]
                print(f"in update_gamma: ms_imgs_q min mean max: {sims_per_pixel.min()}, {sims_per_pixel.mean()}, {sims_per_pixel.max()}")
            sims_per_pixel = sims_per_pixel.flatten()
            sims_per_pixel = torch.sort(sims_per_pixel, descending=False)[0]
            num_pixels_to_reject = int(reject_proportion * sims_per_pixel.numel())
            # find the value of cosine sim that would reject the stated proportion
            # make sure that num_pixels_to_reject is not greater than sims_per_pixel.numel()
            num_pixels_to_reject = min(num_pixels_to_reject, sims_per_pixel.numel()-1)
            cosine_sim_threshold = sims_per_pixel[num_pixels_to_reject]
            gamma_new = cosine_sim_threshold * torch.ones(1, device=self.device).float()
            self.gamma = gamma_new
    ##########################################################################################################################################################


    