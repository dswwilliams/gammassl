import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import wandb
sys.path.append("../")
from datasets.val_datasets import ValDataset
from utils.device_utils import get_lr

class BaseTrainer():
    ##########################################################################################################################################################
    def __init__(self, opt):
        # setting random seed
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.opt = opt

        # class definitions
        self.known_class_list = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        
        # setting up logging of results
        wandb.init(project=self.opt.wandb_project, config=self.opt)
        
        # init
        self.epoch = 0
        self.full_validation_count = 0
        self.ssl_validation_count = 0
        self._init_device()
        self._init_model()
        self._init_validation()
        self._init_training_dataloader()
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def _init_device(self):
        # if available (and not overwridden by opt.use_cpu) use GPU, else use CPU
        if torch.cuda.is_available() and self.opt.use_cpu == False:
            device_id = "cuda:" + self.opt.gpu_no
        else:
            device_id = "cpu"
        
        print("Device: ", device_id)
        self.device = torch.device(device_id)
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def _init_model(self):
        from models.model import SegmentationModel
        self.model = SegmentationModel(device=self.device, opt=self.opt, known_class_list=self.known_class_list)
        if self.opt.save_path is not None:
            checkpoint = torch.load(self.opt.save_path, map_location=self.device)
            a = self.model.seg_net.backbone.load_state_dict(checkpoint['backbone'], strict=False)
            missing_keys, unexpected_keys = a.missing_keys, a.unexpected_keys
            print("Missing keys: ", missing_keys)
            print("Unexpected keys: ", unexpected_keys)
            self.model.seg_net.decode_head.load_state_dict(checkpoint['decode_head'])
            if self.model.seg_net.seg_head is not None:
                self.model.seg_net.seg_head.load_state_dict(checkpoint['seg_head'])
            if self.model.seg_net.projection_net is not None:
                self.model.seg_net.projection_net.load_state_dict(checkpoint['projection_net'])

        self.crop_size = self.model.crop_size
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def _init_training_dataloader(self):
        ##########################################################################################################################################################
        ### setup training dataset ###
        # labelled cityscapes data and unlabelled data from another domain
        from datasets.cityscapes_bdd_dataset import CityscapesxBDDDataset
        _train_dataset = CityscapesxBDDDataset

        if self.opt.no_unlabelled:        # e.g. if labelled cityscapes only
            _only_labelled = True
        else:
            _only_labelled = False

        self.dataset = _train_dataset(
                                    self.opt.cityscapes_dataroot, 
                                    self.opt.unlabelled_dataroot, 
                                    self.opt.no_transforms,
                                    self.opt.min_crop_ratio,
                                    self.opt.max_crop_ratio,
                                    add_resize_noise=self.opt.use_resize_noise,
                                    only_labelled=_only_labelled,
                                    use_imagenet_norm=self.opt.use_imagenet_norm,
                                    no_colour=self.opt.no_colour,
                                    crop_size=self.crop_size,
                                    )

        self.validator.train_seg_idxs = np.random.choice(len(self.dataset), self.opt.n_train_segs, replace=False)

        # define collation function to implement masking if requested
        from utils.collation_utils import get_collate_fn
        if self.opt.mask_input:
            _collate_fn = get_collate_fn(img_size=self.crop_size, patch_size=14, random_mask_prob=self.opt.random_mask_prob)
        else:
            _collate_fn = None

        self.dataloader = torch.utils.data.DataLoader(
                                                dataset=self.dataset, 
                                                batch_size=self.opt.batch_size, 
                                                shuffle=True, 
                                                num_workers=self.opt.num_workers, 
                                                drop_last=True, 
                                                collate_fn=_collate_fn
                                                )

        self.n_examples = len(self.dataset)
        print(f"No. of training examples per epoch: {self.n_examples}")
        self.its_per_epoch = np.ceil(self.n_examples/self.opt.batch_size)
        print(f"No. of iterations per epoch: {self.its_per_epoch}")
        ##########################################################################################################################################################


    ##########################################################################################################################################################
    def _init_validation(self):
        from validator import Validator
        self.validator = Validator(opt=self.opt, device=self.device, class_labels=self.known_class_list)

        self.val_datasets = []
        cityscapes_val_dataset = ValDataset(
                                        name="CityscapesVal",
                                        dataroot=self.opt.cityscapes_dataroot, 
                                        use_imagenet_norm=self.opt.use_imagenet_norm, 
                                        val_transforms=False,
                                        patch_size=self.model.patch_size,
                                        )
        self.val_datasets.append(cityscapes_val_dataset)
        self.validator.val_seg_idxs[cityscapes_val_dataset.name] = np.random.choice(len(cityscapes_val_dataset), self.opt.n_val_segs, replace=False)

        bdd_val_dataset = ValDataset(
                            name="BDDVal",
                            dataroot=self.opt.bdd_val_dataroot, 
                            use_imagenet_norm=self.opt.use_imagenet_norm, 
                            val_transforms=False,
                            patch_size=self.model.patch_size,
                            )
        self.val_datasets.append(bdd_val_dataset)
        self.validator.val_seg_idxs[bdd_val_dataset.name] = np.random.choice(len(bdd_val_dataset), self.opt.n_val_segs, replace=False)


        for dataset in self.val_datasets:
            n_val_examples = len(dataset)
            print(dataset.name+" - Num. val examples", n_val_examples)
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def train(self):
        # init
        self.it_count = 0
        self.epoch = 0
        ### main training loop ##
        while self.it_count < self.opt.total_iters:
            ### (start of epoch) ###
            for (labelled_dict, raw_dict) in tqdm(self.dataloader):

                ### validation ###
                if (self.it_count % self.opt.val_every == 0):
                    self.model.model_to_eval()
                    if not self.opt.skip_validation:
                        if self.opt.use_proto_seg:
                            self.model.calculate_dataset_prototypes()

                        # log qualitative results
                        self.validator.view_train_segmentations(train_dataset=self.dataset, model=self.model)
                        for dataset in self.val_datasets:
                            self.validator.view_val_segmentations(val_dataset=dataset, model=self.model)
                        wandb.log({"test": 1}, commit=True)

                        # log quantitative results
                        for dataset in self.val_datasets:
                            self.validator.validate_uncertainty_estimation(
                                                                val_dataset=dataset, 
                                                                model=self.model, 
                                                                full_validation_count=self.full_validation_count,
                                                                )
                        if (self.full_validation_count % self.opt.save_every == 0):
                            self.save_model()
                        self.full_validation_count += 1

                ### training ###
                # self.model.model_to_train()
                losses, metrics = self._train_models(labelled_dict, raw_dict)

                self.it_count += 1

                ### log training metrics ###
                data_for_wandb = {}
                if self.it_count % self.opt.log_every == 0:
                    if self.model.gamma is not None:
                        gamma_scalar = self.model.gamma.data.item()
                        data_for_wandb["gamma"] = gamma_scalar

                    for network in self.model.optimizers:
                        data_for_wandb["learning rates/"+str(network)] = get_lr(self.model.optimizers[network])
                    for loss_key in losses:
                        data_for_wandb["losses/"+loss_key] = losses[loss_key]
                    for metric_key in metrics:
                        if "per_class" in metric_key:
                            for k in range(len(self.known_class_list)):
                                data_for_wandb[metric_key+"/"+self.known_class_list[k]] = metrics[metric_key][k]
                        else:
                            data_for_wandb["metrics/"+metric_key] = metrics[metric_key]

                    data_for_wandb["num_training_iterations"] = self.it_count
                    wandb.log(data_for_wandb, commit=True)
            ### (end of epoch) ###
            self.epoch += 1
    ##########################################################################################################################################################


    ##########################################################################################################################################################
    def _train_models(self, labelled_dict, raw_dict):
        return None, None
    ##########################################################################################################################################################


    ######################################################################################################################################################
    def save_model(self):
        SKIP_SAVE = False

        if self.opt.network_destination is not None:
            if not os.path.isdir(self.opt.network_destination):
                os.makedirs(self.opt.network_destination)
        else:
            SKIP_SAVE = True

        if not SKIP_SAVE:
            if self.opt.lora_rank is not None:
                import loralib as lora
                backbone_state_dict = lora.lora_state_dict(self.model.seg_net.backbone)
            else:
                backbone_state_dict = self.model.seg_net.backbone.state_dict()
            save_dict = {
                    "it_count": self.it_count,
                    "backbone": backbone_state_dict,
                    "decode_head": self.model.seg_net.decode_head.state_dict(),
                }
            if self.model.seg_net.seg_head is not None:
                save_dict["seg_head"] = self.model.seg_net.seg_head.state_dict()
            if self.model.seg_net.neck is not None:
                save_dict["neck"] = self.model.seg_net.neck.state_dict()

            if self.model.seg_net.projection_net is not None:
                save_dict["projection_net"] = self.model.seg_net.projection_net.state_dict()
            
            if self.model.dataset_prototypes is not None:
                save_dict["prototypes"] = self.model.dataset_prototypes

            for network in self.model.optimizers:
                save_dict[network+"_optim"] = self.model.optimizers[network].state_dict()
            for network in self.model.schedulers:
                save_dict[network+"_scheduler"] = self.model.schedulers[network].state_dict()

            torch.save(save_dict, os.path.join(self.opt.network_destination, "saved_model_epoch_"+str(self.full_validation_count)+".tar"))
            print("Saved: ", os.path.join(self.opt.network_destination, "saved_model_epoch_"+str(self.full_validation_count)+".tar"))
    ######################################################################################################################################################