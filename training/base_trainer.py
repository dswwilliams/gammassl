import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import wandb
sys.path.append("../")
from utils.device_utils import get_lr, init_device
from utils.disk_utils import load_checkpoint_if_exists, get_encoder_state_dict

class BaseTrainer():
    
    
    def __init__(self, opt):
        self.RANDOM_SEED = 0
        self.opt = opt
        self.set_random_seed()
        self.known_class_list = self.get_known_classes()
        self.init_logging()

        self.device = self.init_device()
        self.model = self.init_model()
        self.validator = self.init_validation()
        self.dataloader = self.init_training_dataloader()
    
    def set_random_seed(self):
        torch.manual_seed(self.RANDOM_SEED)
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)

    def get_known_classes(self):
        return ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    def init_logging(self):
        # only using wandb currently
        if wandb.run is None:
            wandb.init(project=self.opt.wandb_project, config=self.opt)

    def init_device(self):
        return init_device(gpu_no=self.opt.gpu_no, use_cpu=self.opt.use_cpu)
    

    def init_model(self):
        from models.model import SegmentationModel
        model = SegmentationModel(opt=self.opt, known_class_list=self.known_class_list)
        load_checkpoint_if_exists(model, self.opt.save_path)
        return model

    
    def init_training_dataloader(self):
        
        ### setup training dataset ###
        # labelled cityscapes data and unlabelled data from another domain
        # from datasets.cityscapes_bdd_dataset import CityscapesxBDDDataset
        # _train_dataset = CityscapesxBDDDataset

        from datasets.fakedata_dataset import FakeData_Dataset
        _train_dataset = FakeData_Dataset

        if self.opt.sup_loss_only:        # e.g. if labelled cityscapes only
            _only_labelled = True
        else:
            _only_labelled = False

        dataset = _train_dataset(
                                    self.opt.cityscapes_dataroot, 
                                    self.opt.unlabelled_dataroot, 
                                    self.opt.no_transforms,
                                    self.opt.min_crop_ratio,
                                    self.opt.max_crop_ratio,
                                    add_resize_noise=self.opt.use_resize_noise,
                                    only_labelled=_only_labelled,
                                    use_imagenet_norm=self.opt.use_imagenet_norm,
                                    no_colour=self.opt.no_colour,
                                    crop_size=self.model.crop_size,
                                    )

        # define collation function to implement masking if requested
        from utils.collation_utils import get_collate_fn
        if self.opt.mask_input:
            _collate_fn = get_collate_fn(img_size=self.model.crop_size, patch_size=14, random_mask_prob=self.opt.random_mask_prob)
        else:
            _collate_fn = None

        dataloader = torch.utils.data.DataLoader(
                                                dataset=dataset, 
                                                batch_size=self.opt.batch_size, 
                                                shuffle=True, 
                                                num_workers=self.opt.num_workers, 
                                                drop_last=True, 
                                                collate_fn=_collate_fn
                                                )
        return dataloader


    def init_validation(self):
        if not self.opt.skip_validation:
            from ue_testing.tester import Tester
            return Tester(self.opt, self.model)

    def validate_if_needed(self):
        if (self.train_step % self.opt.val_every == 0):
            self.model.model_to_eval()
            if not self.opt.skip_validation:
                if self.opt.use_proto_seg:
                    self.model.calculate_dataset_prototypes()

                # log qualitative results
                self.validator.get_qual_results()

                # log quantitative results
                self.validator.test(self.val_step)

                # save model
                if (self.val_step % self.opt.save_every == 0):
                    self.save_model()
            self.model.model_to_eval()

    def train_model(self, labelled_dict, raw_dict):
        self.train_step += 1
        return NotImplementedError

    def log_metrics(self, losses, metrics):
        log_data = {}
        if self.train_step % self.opt.log_every == 0:
            if self.model.gamma is not None:
                log_data["gamma"] =  self.model.gamma.data.item()

            for network in self.model.optimizers:
                log_data["learning rates/"+str(network)] = get_lr(self.model.optimizers[network])
            for loss_key in losses:
                log_data["losses/"+loss_key] = losses[loss_key]
            for metric_key in metrics:
                if "per_class" in metric_key:
                    for k in range(len(self.known_class_list)):
                        log_data[metric_key+"/"+self.known_class_list[k]] = metrics[metric_key][k]
                else:
                    log_data["metrics/"+metric_key] = metrics[metric_key]

            log_data["num_training_iterations"] = self.train_step
            wandb.log(log_data, commit=True)

    def train(self):
        # init
        self.train_step = 0
        self.val_step = 0
        self.epoch = 0

        while self.train_step < self.opt.num_train_steps:
            for (labelled_dict, raw_dict) in tqdm(self.dataloader):
                
                self.validate_if_needed()
                self.val_step += 1

                losses, metrics = self.train_model(labelled_dict, raw_dict)
                self.train_step += 1

                self.log_metrics(losses, metrics)
            self.epoch += 1
       

    def save_model(self):
        if self.opt.network_destination is None:
            return

        if not os.path.isdir(self.opt.network_destination):
            os.makedirs(self.opt.network_destination)

        encoder_state_dict = get_encoder_state_dict(self.model)
        save_dict = {
                "train_step": self.train_step,
                "val_step": self.val_step,
                "encoder": encoder_state_dict(),
                "decoder": self.model.seg_net.decoder.state_dict(),
            }
        
        if self.model.seg_net.projection_net is not None:
            save_dict["projection_net"] = self.model.seg_net.projection_net.state_dict()
        
        if self.model.dataset_prototypes is not None:
            save_dict["prototypes"] = self.model.dataset_prototypes

        if self.model.gamma is not None:
            save_dict["gamma"] = self.model.gamma.data.item()

        for network in self.model.optimizers:
            save_dict[network+"_optim"] = self.model.optimizers[network].state_dict()
        for network in self.model.schedulers:
            save_dict[network+"_scheduler"] = self.model.schedulers[network].state_dict()

        torch.save(save_dict, os.path.join(self.opt.network_destination, "saved_model_epoch_"+str(self.val_step)+".tar"))
        print("Saved: ", os.path.join(self.opt.network_destination, "saved_model_epoch_"+str(self.val_step)+".tar"))
