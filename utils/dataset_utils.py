import torch
import numpy as np    
import cv2
import torch.nn as nn
import kornia
import random


class ImgColourTransform(nn.Module):
    """
    Class that applies a sequence of random colour transforms to an image.
    """
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

        # option to skip colour transforms
        self.no_colour = no_colour

        if self.no_colour:
            self.transform_list = [sharpen, gaussian_noise]
        else:
            self.transform_list = [sharpen, posterize, random_equalize, motion_blur, grayscale, gaussian_noise, channel_shuffle]


    def forward(self, img):
        # choose transforms from list
        transforms = random.sample(self.transform_list, k=self.k)
        if not self.no_colour:
            transforms.append(self.jitter)

        # img needs to be in range: [0,1]
        if (img > 2).any():
            img = img/255
        for transform in transforms:
            img = torch.clamp(transform(img.float()).float(), 0, 1)
        return img.squeeze()
    

def get_resize_noise(height, crop_size, noise_factor):
    """
    - we want to control ratio  = (height/crop_size)
    - we want noise range: [ratio - alpha , ratio + alpha]
    - and also ratio > 1, to prevent bad crops
    """
    noise = 2 * (np.random.rand() - 0.5)        # range: [-1,1]
    ratio = (height/crop_size)        # how many times bigger is shorter side than crop_size 

    alpha = (ratio - 1) / noise_factor
    noise = ratio + noise * alpha         # range: [ratio - alpha , ratio + alpha]

    return noise
    

def get_random_crop(img, label=None, crop_size=256, start_x=None, start_y=None):
    """
    Applies a random crop to the image and label (if provided).
    The size of the crop is defined by crop_size, and the starting position is randomly chosen.
    """
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

def get_initial_scaling_values(height, width, downsample_factor):
    """
    Returns dimensions for imgs after initial scaling.
    Applied to all imgs in Dataset classes.
    """
    scale_factor = height/960
    new_height = int(height/(downsample_factor*scale_factor))
    new_width = int(width/(downsample_factor*scale_factor))
    return new_height, new_width


def resize_data(img, new_height, new_width, label=None):
    """
    Applies resizing to the image and label (if provided).
    Based on the new_height and new_width.
    """
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    if label is not None:
        label = cv2.resize(label, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return img, label
    

def random_flip(img, label=None, p=0.5):
    """
    Applies a random flip to the image and label (if provided).
    The flip is applied with probability p.
    """
    if np.random.rand() < p:
        img = cv2.flip(img, 1)
        if label is not None:
            label = cv2.flip(label, 1)

    # return img, label or img, None
    return img, label

def normalize_img(img):
    """
    Normalizes the image based on ImageNet statistics.
    """
    # normalize the img (with the mean and std for the pretrained ResNet):
    if (img > 2).any():
        img = img/255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
    img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
    img = img.astype(np.float32)
    return img

def normalize_img_tensor(img, imagenet=False):
    """
    Normalizes an image tensor based on ImageNet statistics or to [-1, 1].
    """
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

def central_crop_img(img, output_shape, label=None):
    """
    Applies a central crop to the image and label (if provided).
    The size of the central crop is defined by output_shape.
    """
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
    Determines the dimensions the image should be resized to and cropped to based on the aspect ratio.
    If patch_size is not None, the width is set to be divisible by 2*patch_size.
    This is required for the ViT-type encoders.    
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