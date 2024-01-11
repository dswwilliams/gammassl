import os
import sys
import cv2
import numpy as np
import torch

from utils.dataset_utils import get_initial_scaling_values, central_crop_img
from utils.colour_transforms import ImgColourTransform

cityscapes_train_dirs = ["train/jena/", "train/zurich/", "train/weimar/", "train/ulm/", "train/tubingen/", "train/stuttgart/",
              "train/strasbourg/", "train/monchengladbach/", "train/krefeld/", "train/hanover/",
              "train/hamburg/", "train/erfurt/", "train/dusseldorf/", "train/darmstadt/", "train/cologne/",
              "train/bremen/", "train/bochum/", "train/aachen/"]
cityscapes_val_dirs = ["val/frankfurt/", "val/munster/", "val/lindau/"]
cityscapes_test_dirs = ["test/berlin/", "test/bielefeld/", "test/bonn/", "test/leverkusen/", "test/mainz/", "test/munich/"]

DOWNSAMPLE_FACTOR = 2

def normalize_img(img, imagenet=False):
    if imagenet:
        # normalize the img (with the mean and std for the pretrained ResNet):
        if (img > 2).any():
            img = img/255.0
        img = img - torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        img = img/torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2) # (shape: (256, 256, 3))
        img = img.float()
    else:
        # normalize the img (with the mean and std for the pretrained ResNet):
        if (img > 2).any():
            img = img/255.0
        # [0,1] -> [-1, 1]
        img = (img - 0.5) * 2
        img = img.float()
    return img

# def get_initial_scaling_values(height, width, downsample_factor):
#     scale_factor = height/960
#     new_height = int(height/(downsample_factor*scale_factor))
#     new_width = int(width/(downsample_factor*scale_factor))
#     return new_height, new_width

def get_preprocessed_data(example, imagenet_norm, downsample_factor, use_dino=False, val_transforms=False, colour_transform=None, use_dinov1=False):
    img = cv2.cvtColor(cv2.imread(example["img_path"]), cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    label_img = cv2.imread(example["label_path"], cv2.IMREAD_GRAYSCALE)
    new_img_h, new_img_w = get_initial_scaling_values(h, w, downsample_factor)

    if use_dinov1:
        new_img_h = int(new_img_h/32)*32                    # div by 32
        new_img_w = int(np.round(new_img_h * (w/h)))        # not div by 32, but kept aspect ratio
    elif use_dino:
        new_img_h = int(new_img_h/28)*28                    # div by 28
        new_img_w = int(np.round(new_img_h * (w/h)))        # not div by 28, but kept aspect ratio
        
    img = cv2.resize(img, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST)
    label_img = cv2.resize(label_img, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST)

    if use_dinov1:
        new_img_w_div_28 = int(new_img_w/32)*32             # div by 32
        img, label_img = central_crop_img(img, label=label_img, output_shape=(new_img_h, new_img_w_div_28))
    elif use_dino:
        # central crop to make new_img_w divisible by 28
        new_img_w_div_28 = int(new_img_w/28)*28             # div by 28
        img, label_img = central_crop_img(img, label=label_img, output_shape=(new_img_h, new_img_w_div_28))

    ### converting numpy -> torch ###
    img = torch.from_numpy(img) 
    img = img.permute(2,0,1).float()/255
    if val_transforms:
            img = colour_transform(img)
    img = normalize_img(img, imagenet=imagenet_norm)
    label_img = torch.from_numpy(label_img).long()

    return img, label_img


class CityscapesValDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, use_dino, use_imagenet_norm, val_transforms=False, use_dinov1=False):
        self.name = "CityscapesVal"
        self.cityscapes_img_dir = os.path.join(dataroot, "cityscapes/leftImg8bit/")
        self.cityscapes_label_dir = os.path.join(dataroot, "cityscapes/meta/label_imgs/")
        self.imagenet_norm = use_imagenet_norm
        self.downsample_factor = DOWNSAMPLE_FACTOR

        self.use_dino = use_dino
        self.use_dinov1 = use_dinov1

        self.val_transforms = val_transforms

        self.colour_transform = ImgColourTransform(n_seq_transforms=1)
        
        ### FINDING CITYSCAPES DATA ###
        self.examples = []
        for val_dir in cityscapes_val_dirs:
            val_img_dir_path = self.cityscapes_img_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]
                img_path = val_img_dir_path + file_name
                label_img_path = self.cityscapes_label_dir + img_id + ".png"
                example = {}
                example["name"] = "cityscapes"
                example["img_path"] = img_path
                example["label_path"] = label_img_path
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img, label_img = get_preprocessed_data(example, self.imagenet_norm, self.downsample_factor, self.use_dino, self.val_transforms, self.colour_transform, self.use_dinov1)

        output = {}
        output["img"] = img
        output["label"] = label_img
        return output

    def __len__(self):
        return self.num_examples