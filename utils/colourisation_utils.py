import torch
import torch.nn.functional as F
import numpy as np
import kornia
import matplotlib.pyplot as plt


def denormalise(batch, imagenet=False):
    if imagenet:
        # Define the normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)

        # Denormalize the batch
        denorm_batch = batch * std + mean
        denorm_batch = torch.clamp(denorm_batch * 255, min=0, max=255)

    else:
        # [-1, 1] -> [-0.5, 0.5] -> [0, 1] -> [0,255]
        denorm_batch = 255 * (0.5 * batch + 0.5)

    return denorm_batch    


def colorize_segmentations(trainid_img):
    """
    Convert trainid_img to a color image using a predefined color map.
    trainid_img: torch.Tensor with shape (batch_size, height, width) which contains integer class value as pixel values.
    """
    # init
    label_to_color = torch.tensor([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]
    ]).float().to(trainid_img.device)
    trainid_img = trainid_img.long()
    bs, h, w = trainid_img.shape    
    img_colour = torch.zeros(bs, 3, h, w, device=trainid_img.device)
    
    # map each label to a color
    img_colour = label_to_color[trainid_img].permute(0, 3, 1, 2)  # Permute to match [bs, 3, h, w]
    
    return img_colour


def make_overlay(imgs, imagenet):
    imgs = denormalise(imgs, imagenet)/255
    overlays = kornia.color.rgb_to_grayscale(imgs)
    overlays = torch.cat((overlays, overlays, overlays), dim=1)
    return overlays


def get_colourmapped_imgs(grayscale_imgs, cmap=plt.get_cmap('jet')):
    colourmapped_imgs = torch.zeros(grayscale_imgs.shape[0], 3, grayscale_imgs.shape[1], grayscale_imgs.shape[2])
    for batch_no in range(grayscale_imgs.shape[0]):
        colourmapped_imgs[batch_no,:,:,:] = torch.from_numpy(np.delete(cmap(grayscale_imgs[batch_no,:,:].cpu().detach().numpy()), 3, 2)).permute(2,0,1)
    return colourmapped_imgs


def convert_to_vis_seg_masks(seg_masks, overlay=None, interpolation_mode=None, output_size=None):
    if (seg_masks.sum(1).mean() != 1):
        seg_masks = torch.softmax(seg_masks, dim=1)

    if interpolation_mode == "nearest":
        seg_masks = F.interpolate(seg_masks, size=(output_size[0], output_size[1]), mode="nearest")
    elif interpolation_mode == "bilinear":
        seg_masks = F.interpolate(seg_masks, size=(output_size[0], output_size[1]), mode="bilinear", align_corners=True)
    
    vis_ms, vis_seg = torch.max(seg_masks, dim=1)     # shapes: [1, h, w], [1, h, w]

    if overlay is not None:
        vis_seg = 0.5*overlay + 0.5*colorize_segmentations(vis_seg.long().squeeze(1).cpu())/255
    else:
        vis_seg = colorize_segmentations(vis_seg.long().squeeze(1).cpu())/255
    
    if overlay is not None:
        vis_ms = 0.5*overlay + 0.5*get_colourmapped_imgs((1-vis_ms).cpu().squeeze(1).cpu(), cmap=plt.get_cmap("viridis"))
    else:
        vis_ms = get_colourmapped_imgs((1-vis_ms).cpu().squeeze(1).cpu(), cmap=plt.get_cmap("viridis"))

    return vis_seg, vis_ms