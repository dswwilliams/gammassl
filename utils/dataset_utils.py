import torch
import numpy as np    
import cv2

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
