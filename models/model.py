import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device_utils import to_device, init_device
from utils.crop_utils import crop_by_box_and_resize
from utils.downsampling_utils import ClassWeightedModalDownSampler
from tqdm import tqdm
import numpy as np
import copy
import sys
sys.path.append("../")
from utils.hypersphere_prototype_utils import extract_prototypes, segment_via_prototypes


class SegmentationModel(nn.Module):
    """
    class that deals with defining and updating: neural networks, prototypes and gamma 
    
    TODO: make sure that the methods in this class are generic to any defined model, i.e. vit or resnet
          i.e. they call generic functions from the self.seg_net
    """
    
    ##########################################################################################################################################################
    def __init__(self, opt, known_class_list, training_dataset, validation_dataset):
        super().__init__()
        self.opt = opt
        self.num_known_classes = len(known_class_list)

        self.device = init_device(gpu_no=self.opt.gpu_no, use_cpu=self.opt.use_cpu)
        self.seg_net, self.crop_size, self.patch_size = self.init_seg_net()
        self.target_seg_net = self.init_target_seg_net()
        self.gamma = self.init_gamma()
        self.dataset_prototypes = self.init_dataset_prototypes()
        self.class_weighted_modal_downsampler = ClassWeightedModalDownSampler(known_class_list)
        self.init_prototype_dataloaders(training_dataset, validation_dataset)
        self.optimizers = self.init_optimizers()
        self.schedulers = self.init_schedulers()

        # TODO
        self.batch_prototypes = None
        self.old_prototypes = None

    def init_seg_net(self):
        # determine model architecture, and corresponding crop_size
        if self.opt.model_arch == "vit_m2f":
            from models.vit_m2f_seg_net import ViT_M2F_SegNet as SegNet
            crop_size = 224
        elif self.opt.model_arch == "deeplab":
            from models.deeplab_seg_net import DeepLabSegNet as SegNet
            crop_size = 256

        # init seg_net
        seg_net = SegNet(self.device, self.opt, num_known_classes=self.num_known_classes)
        if self.opt.lora_rank is not None:
            import loralib as lora
            lora.mark_only_lora_as_trainable(seg_net.encoder)

        # get patch_size if it exists
        if hasattr(seg_net.encoder, "patch_size"):
            patch_size = seg_net.encoder.patch_size
        else:
            patch_size = None

        return seg_net, crop_size, patch_size
    
    def init_target_seg_net(self):
        """
        TODO: load in network weights along with loading in weights of normal seg_net
        """
        if self.opt.frozen_target:
            target_seg_net = copy.deepcopy(self.seg_net)
            target_seg_net.to(self.device)
            
            if self.opt.frozen_target_save_path:
                print("loading frozen target from ->", self.opt.frozen_target_save_path)
                checkpoint = torch.load(self.opt.frozen_target_save_path, map_location=self.device)
            elif self.opt.save_path:
                print("loading frozen target from ->", self.opt.save_path)
                checkpoint = torch.load(self.opt.save_path, map_location=self.device)

            if self.opt.frozen_target_save_path or self.opt.save_path:
                target_seg_net.encoder.load_state_dict(checkpoint["encoder"], strict=False)
                target_seg_net.decoder.load_state_dict(checkpoint["decoder"])
        else:
            target_seg_net = None

        return target_seg_net
    
    def init_gamma(self):
        return torch.zeros(1, dtype=torch.float32).to(self.device)
    
    def init_dataset_prototypes(self):
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

    def init_prototype_dataloaders(self, training_dataset, validation_dataset):
        train_proto_dataset = copy.deepcopy(training_dataset)
        train_proto_dataset.only_labelled = True
        self.train_proto_dataloader = torch.utils.data.DataLoader(dataset=train_proto_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, drop_last=True)
        self.train_proto_iterator = iter(self.train_proto_dataloader)

        self.val_proto_dataset = copy.deepcopy(validation_dataset)
        self.val_proto_dataset.only_labelled = True
        self.val_proto_dataset.no_appearance_transform = True
        self.val_proto_dataset.add_resize_noise = False
        self.val_proto_dataset.no_colour = True
        self.val_proto_dataset.crop_size = self.crop_size   

    def init_optimizers(self):
        _optimizer = torch.optim.AdamW
        encoder_lr = self.opt.lr_encoder if self.opt.lr_encoder is not None else self.opt.lr

        # setup optmisers for each model part 
        optimizers = {}

        if self.opt.lora_rank is not None:
            encoder_trainable_params = []
            for name, param in self.seg_net.encoder.named_parameters():
                if "lora" in name:
                    encoder_trainable_params.append(param)

            optimizers["encoder"] = _optimizer(
                                            params=encoder_trainable_params, 
                                            lr=encoder_lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),
                                            )
        else:
            optimizers["encoder"] = _optimizer(
                                            params=list(self.seg_net.encoder.parameters()), 
                                            lr=encoder_lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),
                                            )

        optimizers["decoder"] = _optimizer(
                                            params=list(self.seg_net.decoder.parameters()), 
                                            lr=self.opt.lr, 
                                            weight_decay=self.opt.model_weight_decay,
                                            betas=(0.9, 0.999),
                                            )

        if self.seg_net.projection_net is not None:
            optimizers["projection_net"] = _optimizer(
                                        params=list(self.seg_net.projection_net.parameters()), 
                                        lr=self.opt.lr, 
                                        weight_decay=self.opt.model_weight_decay,
                                        betas=(0.9, 0.999),
                                        )
        return optimizers


    def init_schedulers(self):
        # setup learning rate scheduling
        schedulers = {}
        if self.opt.lr_policy == None:
            pass
        elif self.opt.lr_policy == "poly":
            for network in self.optimizers:
                schedulers[network] = torch.optim.lr_scheduler.PolynomialLR(self.optimizers[network], power=1.0, total_iters=self.opt.num_train_steps)

        elif self.opt.lr_policy == "warmup_poly":
            if (self.opt.warmup_ratio is not None) and (self.opt.n_warmup_iters) is not None:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizers[network], start_factor=self.opt.warmup_ratio, end_factor=1, total_iters=self.opt.n_warmup_iters)
                decay_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizers[network], power=1.0, total_iters=self.opt.num_train_steps)
                schedulers[network] = torch.optim.lr_scheduler.SequentialLR(self.optimizers[network], [warmup_scheduler, decay_scheduler], milestones=[self.opt.n_warmup_iters])
            else:
                raise ValueError("Warmup policy selected but warmup parameters not specified")
        return schedulers

    def model_to_train(self):
        self.seg_net.train()

    def model_to_eval(self):
        self.seg_net.eval()


    def get_seg_masks(self, imgs, high_res=False, masks=None, return_mask_features=False, branch=None):
        """
        Options:
        - branch = "query" and use_proto_seg = True
        - branch = "query" and use_proto_seg = False
        - branch = "target" and frozen_target = True
        - branch = "target" and frozen_target = False
        """

        if branch == "target" and self.target_seg_net:
            return self.target_seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks, return_mask_features=return_mask_features)
        elif branch == "target" and not self.target_seg_net:
            return self.seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks, return_mask_features=return_mask_features)
        elif branch == "query" and self.opt.use_proto_seg:
            return self.proto_segment_imgs(imgs, use_dataset_prototypes=False, output_spread=False, include_void=False, masks=masks)
        elif branch == "query" and not self.opt.use_proto_seg:
            return self.seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks, return_mask_features=return_mask_features)
        else:
            return self.seg_net.get_seg_masks(imgs, high_res=high_res, masks=masks, return_mask_features=return_mask_features)


    def get_val_seg_masks(self, imgs):
        """
        - want to validate both query and target branches
        """
        if self.opt.use_proto_seg:
            seg_masks_K_q = self.proto_segment_imgs(imgs, use_dataset_prototypes=True)
        else:
            seg_masks_K_q = self.get_seg_masks(imgs, high_res=True, branch="query")
        segs_K_q = torch.argmax(seg_masks_K_q, dim=1)
        ms_imgs_q = torch.max(seg_masks_K_q, dim=1)[0]
        uncertainty_maps_q = 1 - ms_imgs_q
        query = {"segs": segs_K_q, "uncertainty_maps": uncertainty_maps_q}

        seg_masks_K_t = self.get_seg_masks(imgs, high_res=True, branch="target")    
        segs_K_t = torch.argmax(seg_masks_K_t, dim=1)
        ms_imgs_t = torch.max(seg_masks_K_t, dim=1)[0]
        uncertainty_maps_t = 1 - ms_imgs_t
        target = {"segs": segs_K_t, "uncertainty_maps": uncertainty_maps_t}

        return {"query":query, "target":target}
        
    def proto_segment_features(self, features, img_spatial_dims=None, use_dataset_prototypes=False, include_void=False):
        if use_dataset_prototypes:
            prototypes = self.dataset_prototypes
        else:
            prototypes = self.batch_prototypes
        
        proj_features = self.seg_net.projection_net(features)

        if include_void:
            _gamma = self.gamma
        else:
            _gamma = None

        seg_masks_q_tB = segment_via_prototypes(
                                            proj_features,
                                            prototypes.detach(),         # NB: to prevent backprop through them for this task
                                            gamma=_gamma,
                                            )
        if img_spatial_dims is not None:
            H,W = img_spatial_dims
            seg_masks_q_tB = F.interpolate(seg_masks_q_tB, size=(H,W), mode="bilinear", align_corners=True)
        return seg_masks_q_tB

    def proto_segment_imgs(self, imgs, use_dataset_prototypes=False, output_spread=False, include_void=False, masks=None):
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
        seg_masks_q_tB, mean_sim_to_NNprototype_q = segment_via_prototypes(
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
        labelled_features_A = self.seg_net.extract_proj_features(labelled_imgs_A)


        # upsample labels to multiple of feature dimension in order to use class_weighted_modal_downsampling (it needs an integer downsmple factor)
        feature_spatial_dim = labelled_features_A.shape[-1]
        labels_spatial_dim = labels_A.shape[-1]
        new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
        labels_A = F.interpolate(labels_A.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
        low_res_labels_A = self.class_weighted_modal_downsampler(labels_A, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)

        prototypes = extract_prototypes(labelled_features_A, low_res_labels_A, output_metrics=False)

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
                    low_res_labels = self.class_weighted_modal_downsampler(labels, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)
                else:
                    # NOTE this is SOOO small (even smaller than patch embeddings)
                    feature_spatial_dim = 8
                    labels_spatial_dim = labels.shape[-1]
                    new_labels_spatial_dim = int(np.ceil(labels_spatial_dim/feature_spatial_dim) * feature_spatial_dim)
                    labels = F.interpolate(labels.unsqueeze(1).float(), size=(new_labels_spatial_dim, new_labels_spatial_dim), mode="nearest").squeeze(1).long()
                    low_res_labels = self.class_weighted_modal_downsampler(labels, downsample_factor=new_labels_spatial_dim // feature_spatial_dim)


                labelled_features = self.seg_net.extract_proj_features(labelled_imgs)
                prototypes = extract_prototypes(labelled_features, low_res_labels, output_metrics=False)

                prototypes_sum += prototypes

                if step > 500:
                    break

            prototypes = F.normalize(prototypes_sum, dim=0, p=2)          # shape: [feature_length, num_known_classes]

            self.dataset_prototypes = prototypes
            return self.dataset_prototypes
        else:
            return None
        
    ##########################################################################################################################################################
    def update_gamma(self, seg_masks_q, seg_masks_t):

        segs_q, segs_t = seg_masks_q.detach().argmax(1), seg_masks_t.detach().argmax(1)
        consistency_masks = torch.eq(segs_q, segs_t).float()
        mean_consistency = consistency_masks.mean()

        with torch.no_grad():
            ### calculate gamma so that p(certain) = p(consistent) ###
            reject_proportion = (1 - mean_consistency)   # reject (1 - estimated accuracy)
            # NOTE: think about what scores we are giving it
            # so gamma_scaling determines whether gamma is calculated from the cosine similarity scores or the softmax scores
            if (self.opt.gamma_scaling is None) or (self.opt.gamma_scaling == "None"):
                sims_per_pixel = torch.max(seg_masks_q.detach(), dim=1)[0]         # shape [bs, h, w]
            elif self.opt.gamma_scaling == "softmax":
                # use temperature weighted softmax scores (like in testing)
                sims_per_pixel = torch.max(torch.softmax(seg_masks_q.detach()/self.opt.gamma_temp, dim=1), dim=1)[0]
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


    
