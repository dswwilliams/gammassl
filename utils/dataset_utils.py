import torch
import numpy as np    
import cv2

import torch.nn as nn
import kornia
import random


class ImgColourTransform(nn.Module):
    def __init__(self, n_seq_transforms, no_colour=False):
        super(ImgColourTransform, self).__init__()
        self.k = n_seq_transforms
        # corruption transforms
        motion_blur = kornia.augmentation.RandomMotionBlur(kernel_size=(3,3), angle=(-30,30),  direction=(-1, 1), p=1.)
        gaussian_noise = kornia.augmentation.RandomGaussianNoise(mean=0, std=0.01, p=1)
        # intensity transforms
        random_equalize = kornia.augmentation.RandomEqualize(p=1)
        sharpen = kornia.augmentation.RandomSharpness(1, p=1)
        # solarize = kornia.augmentation.RandomSolarize(0.1, 0.1, p=1)
        posterize = kornia.augmentation.RandomPosterize(6, p=1)
        # invert = kornia.augmentation.RandomInvert(p=1)
        # colour space transforms
        grayscale = kornia.augmentation.RandomGrayscale(p=1)
        channel_shuffle = kornia.augmentation.RandomChannelShuffle(p=1)
        self.jitter = kornia.augmentation.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=1.)

        self.no_colour = no_colour

        if self.no_colour:
            self.transform_list = [sharpen, gaussian_noise]
        else:
            self.transform_list = [sharpen, posterize, random_equalize, motion_blur, grayscale, gaussian_noise, channel_shuffle]


    def forward(self, img):
        # choose transforms from list
        transforms = random.sample(self.transform_list, k=self.k)
        if self.no_colour:
            pass
        else:
            transforms.append(self.jitter)
        # img needs to be in range: [0,1]
        if (img > 2).any():
            img = img/255
        for transform in transforms:
            img = torch.clamp(transform(img.float()).float(), 0, 1)
        return img.squeeze()
    

def get_random_crop(img, label=None, crop_size=256, start_x=None, start_y=None):
    if start_x is None:
        start_x = np.random.randint(low=0, high=(img.shape[1] - crop_size))
    end_x = start_x + crop_size
    if start_y is None:
        start_y = np.random.randint(low=0, high=(img.shape[0] - crop_size))
    end_y = start_y + crop_size

    crop = img[start_y:end_y, start_x:end_x] # shape: (crop_size, crop_size, 3)
    
    if label is not None:
        label = label[start_y:end_y, start_x:end_x] # shape: (crop_size, crop_size)
        return crop, label
    else:
        return crop

def get_initial_scaling_values(height, width, downsample_factor):
    scale_factor = height/960
    new_height = int(height/(downsample_factor*scale_factor))
    new_width = int(width/(downsample_factor*scale_factor))
    return new_height, new_width
    

def random_flip(img, label=None, p=0.5):
    if np.random.rand() < p:
        img = cv2.flip(img, 1)
        if label is not None:
            label = cv2.flip(label, 1)

    if label is not None:
        return img, label
    else:
        return img

def normalize_img(img):
    # normalize the img (with the mean and std for the pretrained ResNet):
    if (img > 2).any():
        img = img/255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
    img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
    img = img.astype(np.float32)
    return img

def normalize_img_tensor(img, imagenet=False):
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

def get_random_crop_from_centre(img, label=None, crop_size=256, crop_centre=None):
    if crop_centre is None:
        start_x = np.random.randint(low=0, high=(img.shape[1] - crop_size))
        start_y = np.random.randint(low=0, high=(img.shape[0] - crop_size))

    else:
        start_x = int(crop_centre[0] - np.ceil(crop_size//2))
        start_y = int(crop_centre[1] - np.ceil(crop_size//2))

    end_x = start_x + crop_size
    end_y = start_y + crop_size

    crop = img[start_y:end_y, start_x:end_x] # shape: (crop_size, crop_size, 3)
    
    if label is not None:
        label = label[start_y:end_y, start_x:end_x] # shape: (crop_size, crop_size)
        return crop, label
    else:
        return crop

def central_crop_img(img, output_shape, label=None):
    h, w, _ = img.shape
    new_h, new_w = output_shape
    h_start = int((h - new_h)/2)
    w_start = int((w - new_w)/2)
    if label is None:
        return img[h_start:h_start+new_h, w_start:w_start+new_w, :]
    else:
        return img[h_start:h_start+new_h, w_start:w_start+new_w, :], label[h_start:h_start+new_h, w_start:w_start+new_w]

def get_img_size_from_aspect_ratio(aspect_ratio, patch_size=None):
    """
    aspect_ratio = (H,W) or is proportional to this
    - preprocessing of imgs is a resize to the right scale (adjusting for patch size and keeping aspect ratio)
    - then if required, a crop such that width is also divisible by patch size
    """
    if patch_size is None:
        # now 2*patch_size and therefore patch size has no effect
        patch_size = 0.5

    # height of imgs are set to 480, width is scaled accordingly
    new_H = 480
    # set width to be divisible by 2*patch_size
    resize_H = int(new_H/(2*patch_size)) * int(2*patch_size)
    # scale width accord to aspect ratio
    resize_W  = int(np.round(resize_H * (aspect_ratio[1]/aspect_ratio[0])))
    resize_sizes = (resize_H, resize_W)

    # set width to be divisible by patch_size
    crop_W = int(resize_W/(2*patch_size)) * int(2*patch_size)
    crop_H = resize_H
    crop_sizes = (crop_H, crop_W)

    return resize_sizes, crop_sizes