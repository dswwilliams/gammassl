import torch
import numpy as np
import os
import sys
import csv
sys.path.append("../")
from utils.candr_utils import get_random_crop_boxes

### definitions ###
DOWNSAMPLE_FACTOR = 2.7273
# CROP_SIZE = 208
RESIZE_NOISE_FACTOR = 1.5


class FakeData_Dataset(torch.utils.data.Dataset):
    def __init__(self, labelled_dataroot, 
                        raw_dataroot, 
                        no_appearance_transform=False, 
                        min_crop_ratio=1.2, 
                        max_crop_ratio=3, 
                        add_resize_noise=True,
                        only_labelled=False,
                        use_imagenet_norm=True,
                        no_colour=False,
                        crop_size=208,
                        ):
        self.big_crop_size = crop_size
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.only_labelled = only_labelled

    def __getitem__(self, index):
        ########################################################################################################################

        labelled_dict = {}
        if np.random.rand() > 0.5:
            labelled_dict["box_A"] = get_random_crop_boxes(
                                                input_size=(self.big_crop_size, self.big_crop_size), 
                                                min_crop_ratio=self.min_crop_ratio, 
                                                max_crop_ratio=self.max_crop_ratio,
                                                )
        else:
            labelled_dict["box_A"] = torch.tensor([[0,0], [self.big_crop_size-1, 0], [self.big_crop_size-1, self.big_crop_size-1], [0, self.big_crop_size-1]])
        
        
        labelled_dict["img"] = torch.randn(3,self.big_crop_size,self.big_crop_size)
        labelled_dict["label"] = torch.randint(0, 19, (self.big_crop_size, self.big_crop_size))

        ########################################################################################################################
        if not self.only_labelled:
            ########################################################################################################################
            ### reading in raw_img ###
            raw_dict = {}

            raw_dict["img_1"] = torch.randn(3,self.big_crop_size,self.big_crop_size)
            raw_dict["img_2"] = torch.randn(3,self.big_crop_size,self.big_crop_size)

            if np.random.rand() > 0.5:
                raw_dict["box_A"] = get_random_crop_boxes(input_size=(self.big_crop_size, self.big_crop_size), 
                                                            min_crop_ratio=self.min_crop_ratio, max_crop_ratio=self.max_crop_ratio)
                raw_dict["box_B"] = torch.tensor([[0,0], [self.big_crop_size-1, 0], [self.big_crop_size-1, self.big_crop_size-1], [0, self.big_crop_size-1]])
            else:
                raw_dict["box_A"] = torch.tensor([[0,0], [self.big_crop_size-1, 0], [self.big_crop_size-1, self.big_crop_size-1], [0, self.big_crop_size-1]])
                raw_dict["box_B"] = get_random_crop_boxes(input_size=(self.big_crop_size, self.big_crop_size), 
                                                            min_crop_ratio=self.min_crop_ratio, max_crop_ratio=self.max_crop_ratio)
            ########################################################################################################################

            return (labelled_dict, raw_dict)
        else:
            return labelled_dict, {}


    def __len__(self):
        return 1000
