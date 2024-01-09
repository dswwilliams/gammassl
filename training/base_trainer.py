import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import visdom
import random
import wandb
sys.path.append("../")
from datasets.cityscapes_val import CityscapesValDataset
from datasets.sax_test_datasets import SAXLondonDataset, SAXNewForestDataset, SAXScotlandDataset
from utils.device_utils import to_device, get_lr
from utils.validation_utils import init_val_ue_metrics, perform_batch_ue_validation, update_running_variable, plot_val_ue_metrics_to_tensorboard


class BaseTrainer():
    ##########################################################################################################################################################
    def __init__(self, opt):
        ### setting random seed ###
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.opt = opt
        self.epoch = 0
        self.full_validation_count = 0
        self.ssl_validation_count = 0
        self.best_ssl_auc = 0


        # class definitions
        if self.opt.sunrgbd:
            self.known_class_list = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", 
                "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor_mat", "clothes", "ceiling", "books", "fridge", 
                "tv", "paper", "towel", "shower_curtain", "box", "whiteboard", "person", "night_stand", "toilet", "sink", "lamp", "bathtub", "bag"]
        elif self.opt.scannet:
            self.known_class_list = ['Bed','Books','Ceiling','Chair','Floor','Furniture','Objects','Picture','Sofa','Table','TV','Wall','Window']
        else:
            self.known_class_list = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        
        if (self.opt.sax_raw_dataroot.split("/")[1] == "jmain02"):
            os.environ["WANDB_MODE"] = "offline"
        
        # setting up results logging
        if self.opt.use_wandb:
            wandb.init(project="gammassl", config=self.opt)
            self.writer = None
        else:
            if self.opt.log_dir is not None:
                self.writer = SummaryWriter(log_dir=self.opt.log_dir)
            elif self.opt.exp_name is not None:
                self.writer = SummaryWriter(log_dir=os.path.join("..", "training", "runs", self.opt.exp_name))
            else:
                self.writer = SummaryWriter()   # for tensorboard
            self.writer.add_text("opt", str(opt))

        if self.opt.output2visdom:
            self.vis = visdom.Visdom(port=self.opt.visdom_port)
            env_list = self.vis.get_env_list()
            self.vis.fork_env(env_list[0], "qualitative-seg-results")
        else:
            self.vis = None

        self._init_device()
        self._init_model()
        self._init_validation()
        self._init_training_dataloader()
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def _init_device(self):
        if self.opt.use_cpu == False:
            device_id = 'cuda:' + self.opt.gpu_no
        else:
            device_id = 'cpu'
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
    ##########################################################################################################################################################

    ##########################################################################################################################################################
    def _init_training_dataloader(self):
        ##########################################################################################################################################################
        ### setup training dataset ###
        if self.opt.sunrgbd:
            from datasets.sunrgbd_dataset import SUNRGBD_Dataset
            _train_dataset = SUNRGBD_Dataset
            _only_labelled = True
            self.dataset = _train_dataset(
                                        self.opt.sunrgbd_dataroot,
                                        "train",
                                        self.opt.no_transforms,
                                        self.opt.min_crop_ratio,
                                        self.opt.max_crop_ratio,
                                        add_resize_noise=self.opt.use_resize_noise,
                                        only_labelled=_only_labelled,
                                        use_imagenet_norm=self.opt.use_imagenet_norm,
                                        )
        elif self.opt.scannet:
            if self.opt.no_unlabelled:
                from datasets.scannet_dataset import ScanNetDataset
                _train_dataset = ScanNetDataset
                _only_labelled = True
                self.dataset = _train_dataset(
                                            self.opt.scannet_dataroot,
                                            "train",
                                            self.opt.no_transforms,
                                            self.opt.min_crop_ratio,
                                            self.opt.max_crop_ratio,
                                            add_resize_noise=self.opt.use_resize_noise,
                                            only_labelled=_only_labelled,
                                            use_imagenet_norm=self.opt.use_imagenet_norm,
                                            )
            
            else:
                print("Training on ScanNet and ImageNet")
                if not self.opt.use_scannet_twice:
                    from datasets.imagenet_dataset import ScanNet_Imagenet_Dataset
                    _train_dataset = ScanNet_Imagenet_Dataset
                    self.dataset = _train_dataset(
                                            imagenet_dataroot=self.opt.imagenet_dataroot,
                                            scannet_dataroot=self.opt.scannet_dataroot,
                                            no_appearance_transform=self.opt.no_transforms,
                                            min_crop_ratio=self.opt.min_crop_ratio,
                                            max_crop_ratio=self.opt.max_crop_ratio,
                                            add_resize_noise=self.opt.use_resize_noise,
                                            only_labelled=False,
                                            use_imagenet_norm=self.opt.use_imagenet_norm,
                                            use_dino=self.opt.use_dino,
                                            )
                else:
                    print("Training on ScanNet twice")
                    from datasets.scannet_scannet_dataset import ScanNet_ScanNet_Dataset
                    _train_dataset = ScanNet_ScanNet_Dataset
                    self.dataset = _train_dataset(
                                            scannet_dataroot=self.opt.scannet_dataroot,
                                            no_appearance_transform=self.opt.no_transforms,
                                            min_crop_ratio=self.opt.min_crop_ratio,
                                            max_crop_ratio=self.opt.max_crop_ratio,
                                            add_resize_noise=self.opt.use_resize_noise,
                                            only_labelled=False,
                                            use_imagenet_norm=self.opt.use_imagenet_norm,
                                            use_dino=self.opt.use_dino,
                                            )
        else:
            # labelled cityscapes data and unlabelled SAX data
            print(f"self.opt.use_sax_png_dataset: {self.opt.use_sax_png_dataset}")
            if self.opt.use_sax_png_dataset:
                print("using SAX PNG dataset")
                from datasets.cityscapes_saxpng_dataset import CityscapesxSAXPNGDataset
                _train_dataset = CityscapesxSAXPNGDataset
            elif (self.opt.sax_raw_dataroot.split("/")[1] == "jmain02") or ("/opt/ori/data" in self.opt.sax_raw_dataroot):
                from datasets.cityscapes_saxpng_dataset import CityscapesxSAXPNGDataset
                _train_dataset = CityscapesxSAXPNGDataset
            elif self.opt.no_unlabelled:
                from datasets.cityscapes_only_train import CityscapesOnlyTrainDataset
                _train_dataset = CityscapesOnlyTrainDataset
            else:
                from datasets.cityscapes_mono_datasets import CityscapesxMonolithicDataset
                _train_dataset = CityscapesxMonolithicDataset
            if self.opt.use_fake_data:
                print("using fake data")
                from datasets.fake_dataset import FAKETestDataset
                _train_dataset = FAKETestDataset

            if self.opt.method == "gammassl":
                _only_labelled = False
            else:
                _only_labelled = True
            if self.opt.no_unlabelled:        # overwrite if cityscapes only
                _only_labelled = True

            self.dataset = _train_dataset(
                                        self.opt.cityscapes_dataroot, 
                                        self.opt.sax_raw_dataroot, 
                                        self.opt.sensor_models_path, 
                                        self.opt.sax_domain, 
                                        self.opt.no_transforms,
                                        self.opt.min_crop_ratio,
                                        self.opt.max_crop_ratio,
                                        add_resize_noise=self.opt.use_resize_noise,
                                        only_labelled=_only_labelled,
                                        use_imagenet_norm=self.opt.use_imagenet_norm,
                                        use_dinov1=self.opt.use_dinov1,
                                        no_colour=self.opt.no_colour,
                                        )

        self.validator.train_seg_idxs = np.random.choice(len(self.dataset), self.opt.n_train_segs, replace=False)

        from utils.collation_utils import get_collate_fn

        if self.opt.run_masking_task:
            _collate_fn = get_collate_fn(img_size=224, patch_size=14, random_mask_prob=self.opt.random_mask_prob)
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
        print(self.dataset)
        print("No. of training examples per epoch: ", self.n_examples)
        self.its_per_epoch = np.ceil(self.n_examples/self.opt.batch_size)
        print("No. of iterations per epoch: ", self.its_per_epoch)
        ##########################################################################################################################################################


    ##########################################################################################################################################################
    def _init_validation(self):
        from validator import Validator
        # self.validator = Validator(opt=self.opt, writer=self.writer)
        self.validator = Validator(opt=self.opt, class_labels=self.known_class_list)

        if self.opt.sunrgbd or self.opt.scannet:
            self.val_datasets = []
            from datasets.sunrgbd_dataset import SUNRGBD_Dataset
            if self.opt.scannet:
                use_scannet_classes = True
            else:
                use_scannet_classes = False
            sunrgbd_val_dataset = SUNRGBD_Dataset(self.opt.sunrgbd_dataroot, split="val", use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, scannet_classes=use_scannet_classes)
            self.val_datasets.append(sunrgbd_val_dataset)
            self.validator.val_seg_idxs[sunrgbd_val_dataset.name] = np.random.choice(len(sunrgbd_val_dataset), self.opt.n_val_segs, replace=False)
        else:
            self.val_datasets = []
            cityscapes_val_dataset = CityscapesValDataset(self.opt.cityscapes_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
            self.val_datasets.append(cityscapes_val_dataset)
            self.validator.val_seg_idxs[cityscapes_val_dataset.name] = np.random.choice(len(cityscapes_val_dataset), self.opt.n_val_segs, replace=False)
            if self.opt.sax_domain == "london":
                sax_london_dataset = SAXLondonDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
                self.val_datasets.append(sax_london_dataset)
                self.validator.val_seg_idxs[sax_london_dataset.name] = np.random.choice(len(sax_london_dataset), self.opt.n_val_segs, replace=False)
            elif self.opt.sax_domain == "new-forest":
                # self.val_datasets.append(SAXNewForestDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm))
                sax_new_forest_dataset = SAXNewForestDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
                self.val_datasets.append(sax_new_forest_dataset)
                self.validator.val_seg_idxs[sax_new_forest_dataset.name] = np.random.choice(len(sax_new_forest_dataset), self.opt.n_val_segs, replace=False)
            elif self.opt.sax_domain == "scotland":
                sax_scotland_dataset = SAXScotlandDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
                self.val_datasets.append(sax_scotland_dataset)
                self.validator.val_seg_idxs[sax_scotland_dataset.name] = np.random.choice(len(sax_scotland_dataset), self.opt.n_val_segs, replace=False)

            if self.opt.val_all_sax:
                # london
                sax_london_dataset = SAXLondonDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
                self.val_datasets.append(sax_london_dataset)
                self.validator.val_seg_idxs[sax_london_dataset.name] = np.random.choice(len(sax_london_dataset), self.opt.n_val_segs, replace=False)
                sax_new_forest_dataset = SAXNewForestDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
                # new forest
                self.val_datasets.append(sax_new_forest_dataset)
                self.validator.val_seg_idxs[sax_new_forest_dataset.name] = np.random.choice(len(sax_new_forest_dataset), self.opt.n_val_segs, replace=False)
                # scotland
                sax_scotland_dataset = SAXScotlandDataset(self.opt.sax_labelled_dataroot, use_dino=self.opt.use_dino, use_imagenet_norm=self.opt.use_imagenet_norm, use_dinov1=self.opt.use_dinov1)
                self.val_datasets.append(sax_scotland_dataset)
                self.validator.val_seg_idxs[sax_scotland_dataset.name] = np.random.choice(len(sax_scotland_dataset), self.opt.n_val_segs, replace=False)

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
        # for epoch in range(self.opt.n_epochs):
        while self.it_count < self.opt.total_iters:
            ### (start of epoch) ###
            for (labelled_dict, raw_dict) in tqdm(self.dataloader):

                ### validation ###
                self.model.model_to_eval()
                if (self.it_count % self.opt.full_validation_every == 0):
                    if self.opt.skip_validation:
                        pass
                    else:
                        if self.opt.method == "gammassl" and (self.opt.use_symmetric_branches == False):
                            self.model.calculate_dataset_prototypes()
                        
                        if not (self.opt.sax_raw_dataroot.split("/")[1] == "jmain02"):      # skip if on JADE2
                            # view validation segmentations
                            self.validator.view_train_segmentations(train_dataset=self.dataset, model=self.model)
                            self.validator.view_weird_val_segmentations(opt=self.opt, model=self.model)
                            for dataset in self.val_datasets:
                                self.validator.view_val_segmentations(val_dataset=dataset, model=self.model)
                        wandb.log({"test": 1}, commit=True)

                        for dataset in self.val_datasets:
                            self.validator.validate_uncertainty_estimation(dataset, self.model, self.full_validation_count)
                        if (self.full_validation_count % self.opt.save_every == 0) and not self.opt.validate_only:
                            self.save_model()
                        self.full_validation_count += 1

                if self.opt.validate_only:
                    self.it_count = self.opt.total_iters + 1
                    break


                ### calculate losses and update model ###
                self.model.model_to_train()
                losses, metrics = self._train_models(labelled_dict, raw_dict)

                ### log losses & metrics to tensorboard ###
                self.it_count += 1

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
        elif self.opt.exp_name is not None:
            from os.path import expanduser
            home = expanduser("~")
            self.opt.network_destination = os.path.join(home, "networks", self.opt.exp_name)
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