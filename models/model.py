import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import sys
sys.path.append("../")
from utils.prototype_utils import extract_prototypes, segment_via_prototypes
from utils.disk_utils import load_checkpoint_if_exists
from utils.gamma_utils import calculate_threshold
from utils.device_utils import to_device, init_device
from utils.candr_utils import crop_by_box_and_resize
from utils.downsampling_utils import ClassWeightedModalDownSampler, downsample_labels


class SegmentationModel(nn.Module):
    """
    class that deals with defining and updating: neural networks, prototypes and gamma 
    
    TODO: make sure that the methods in this class are generic to any defined model, i.e. vit or resnet
          i.e. they call generic functions from the self.seg_net
    """
    
    def __init__(self, opt, known_class_list, training_dataset, crop_size):
        super().__init__()
        self.opt = opt
        self.crop_size = crop_size
        self.num_known_classes = len(known_class_list)

        self.device = init_device(gpu_no=self.opt.gpu_no, use_cpu=self.opt.use_cpu)
        self.seg_net, self.patch_size = self.init_seg_net()
        self.target_seg_net = self.init_target_seg_net()
        self.gamma = self.init_gamma()
        self.dataset_prototypes = self.init_dataset_prototypes()
        self.class_weighted_modal_downsampler = ClassWeightedModalDownSampler(known_class_list)
        self.init_prototype_dataloaders(training_dataset)
        self.optimizers = self.init_optimizers()
        self.schedulers = self.init_schedulers()

        # TODO
        self.batch_prototypes = None
        self.old_prototypes = None

    def init_seg_net(self):
        # determine model architecture
        if self.opt.model_arch == "vit_m2f":
            from models.vit_m2f_seg_net import ViT_M2F_SegNet as SegNet
        elif self.opt.model_arch == "deeplab":
            from models.deeplab_seg_net import DeepLabSegNet as SegNet

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

        load_checkpoint_if_exists(seg_net, self.opt.save_path)

        return seg_net, patch_size
    
    def init_target_seg_net(self):
        if self.opt.frozen_target:
            target_seg_net = copy.deepcopy(self.seg_net).to(self.device)
            
            if self.opt.frozen_target_save_path:
                load_checkpoint_if_exists(target_seg_net, self.opt.frozen_target_save_path)
            elif self.opt.save_path:
                load_checkpoint_if_exists(target_seg_net, self.opt.save_path)
            return target_seg_net
        else:
            return None
    
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

    def init_prototype_dataloaders(self, training_dataset):
        self.train_proto_dataset = copy.deepcopy(training_dataset)
        self.train_proto_dataset.only_labelled = True
        self.train_proto_dataset.no_appearance_transform = True
        self.train_proto_dataset.add_resize_noise = False
        self.train_proto_dataset.no_colour = True
        self.train_proto_dataset.crop_size = self.crop_size   
        self.train_proto_dataloader = torch.utils.data.DataLoader(
                                                    dataset=self.train_proto_dataset, 
                                                    batch_size=self.opt.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=self.opt.num_workers, 
                                                    drop_last=True)
        self.train_proto_iterator = iter(self.train_proto_dataloader)

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
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizers[network], 
                                                                        start_factor=self.opt.warmup_ratio, 
                                                                        end_factor=1, 
                                                                        total_iters=self.opt.n_warmup_iters)
                decay_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizers[network], 
                                                                        power=1.0, 
                                                                        total_iters=self.opt.num_train_steps)
                schedulers[network] = torch.optim.lr_scheduler.SequentialLR(self.optimizers[network], 
                                                                            [warmup_scheduler, decay_scheduler], 
                                                                            milestones=[self.opt.n_warmup_iters])
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
        
    def get_next_proto_batch(self, data_iterator, data_loader):
        try:
            data_dict, _ = next(data_iterator)
        except StopIteration:
            # Restart the iterator if it runs out of data
            data_iterator = iter(data_loader)
            data_dict, _ = next(data_iterator)
        except Exception as e:
            # Handle other exceptions if needed
            raise e
        
        return data_dict, data_iterator
    

    def calculate_batch_prototypes(self):
        """ Calculate prototypes from batch of labelled images """

        # reading in data from dedicated dataloader for prototypes
        labelled_dict, self.train_proto_iterator = self.get_next_proto_batch(self.train_proto_iterator, self.train_proto_dataloader)
        labelled_imgs = to_device(labelled_dict["img"], self.device)
        labels = to_device(labelled_dict["label"], self.device)
        labelled_crop_boxes_A = to_device(labelled_dict["box_A"], self.device)
        
        labelled_imgs_A = crop_by_box_and_resize(labelled_imgs, labelled_crop_boxes_A)
        labels_A = crop_by_box_and_resize(labels.unsqueeze(1).float(), labelled_crop_boxes_A, mode="nearest").squeeze(1).long()

        # extract features
        labelled_features_A = self.seg_net.extract_proj_features(labelled_imgs_A)

        # downsample labels to match feature spatial dimensions
        low_res_labels_A = downsample_labels(features=labelled_features_A, labels=labels_A, downsampler=self.class_weighted_modal_downsampler)

        # calculate prototypes
        prototypes = extract_prototypes(labelled_features_A, low_res_labels_A, output_metrics=False)

        # if class doesnt exist in batch, use previous prototype
        for k in range(prototypes.shape[1]):
            proto_for_class = prototypes[:,k]
            # if all values in prototype are zero, there were no features of that class, therefore use previous prototype from that class
            if torch.eq(proto_for_class, 0).all() and self.old_prototypes is not None:
                prototypes[:,k] = self.old_prototypes[:,k].to(prototypes.device)

        # assign new batch prototypes to class variable
        self.batch_prototypes = prototypes

        self.old_prototypes = prototypes.detach().cpu()

        return self.batch_prototypes
    
    @torch.no_grad()
    def calculate_dataset_prototypes(self):
        if self.opt.prototypes_path is None:
            """ Calculate prototype from entire dataset """
            dataloader = torch.utils.data.DataLoader(
                                    self.train_proto_dataset, 
                                    batch_size=self.opt.batch_size, 
                                    shuffle=True, 
                                    num_workers=self.opt.num_workers, 
                                    drop_last=True)
            
            iterator = tqdm(dataloader)
            print("calculating dataset prototypes...")
            prototypes_sum = 0
            for labelled_dict,_ in iterator:
                labelled_imgs = labelled_dict["img"].to(self.device)
                labels = labelled_dict["label"].to(self.device) 

                # extract features
                labelled_features = self.seg_net.extract_proj_features(labelled_imgs)

                # downsample labels to match feature spatial dimensions
                low_res_labels = downsample_labels(features=labelled_features, labels=labels, downsampler=self.class_weighted_modal_downsampler)

                # calculate prototypes
                prototypes = extract_prototypes(labelled_features, low_res_labels, output_metrics=False)

                prototypes_sum += prototypes


            prototypes = F.normalize(prototypes_sum, dim=0, p=2)          # shape: [feature_length, num_known_classes]

            self.dataset_prototypes = prototypes
            return self.dataset_prototypes
        else:
            return None
        
    @torch.no_grad()
    def update_gamma(self, seg_masks_q, seg_masks_t):
        """
        Calculate confidence threshold gamma, such that:
            the number of consistent elements between seg_masks_q and seg_masks_t is equal to the number of certain elements of seg_masks_q
        """

        segs_q, segs_t = seg_masks_q.argmax(1), seg_masks_t.argmax(1)
        consistency_masks = torch.eq(segs_q, segs_t)
        num_consistent_pixels = torch.sum(consistency_masks == 0)
        num_inconsistent_pixels = torch.numel(consistency_masks) - num_consistent_pixels

        if self.opt.gamma_scaling == "softmax":
            # gamma is threshold on softmax scores
            confidences = torch.max(torch.softmax(seg_masks_q.detach()/self.opt.gamma_temp, dim=1), dim=1)[0]
        else:
            # gamma is threshold on logits
            confidences = torch.max(seg_masks_q.detach(), dim=1)[0]

        gamma = calculate_threshold(confidences, num_rejects=num_inconsistent_pixels)

        gamma = gamma * torch.ones(1, device=self.device).float()
        self.gamma = gamma


