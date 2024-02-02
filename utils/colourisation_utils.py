import torch
import torch.nn.functional as F
import numpy as np
import kornia
import matplotlib.pyplot as plt

# def denormalise(batch, imagenet=False):
#     if imagenet:
#         # needs to this shape -> (height, width, channels)
#         denorm_batch = torch.zeros_like(batch)
#         batch = batch.detach().cpu().numpy()
#         denorm_batch = denorm_batch.detach().cpu().numpy()

#         batch = np.transpose(batch, (0, 2, 3, 1)) # (batch_idx, height, width, channels)
#         for i in range(0,batch.shape[0]):
#             img = batch[i,:,:,:]
#             img = img*np.array([0.229, 0.224, 0.225])
#             img = img + np.array([0.485, 0.456, 0.406])
#             img = img*255.0
#             img = np.transpose(img, (2, 0, 1)) # (channels, height, width)
#             denorm_batch[i,:,:,:] = img

#         # batch = np.transpose(batch, (0, 3, 1, 2)) # (batch_idx, height, width, channels)
#         denorm_batch = torch.from_numpy(denorm_batch)
#         denorm_batch = torch.clamp(denorm_batch, min=0, max=255)
#         return denorm_batch # denorm_batch tensor has range 0 -> 255
#     else:
#         # [-1, 1] -> [-0.5, 0.5] -> [0, 1] -> [0,255]
#         denorm_batch = 255*(0.5*batch + 0.5)
#         return denorm_batch
    
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
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: (220,220,  0),
        8: [107,142, 35],
        9: [152,251,152] ,
        10: [ 70,130,180],
        11: [220, 20, 60] ,
        12: [255,  0,  0] ,
        13: [  0,  0,142] ,
        14: [  0,  0, 70] ,
        15: [  0, 60,100] ,  
        16: [  0, 80,100] ,
        17: [  0,  0,230] ,
        18: [119, 11, 32] ,
        19: [  0,  0,  0]
        }
    batch_size, img_height, img_width = trainid_img.shape
    img_colour = torch.zeros(batch_size, 3, img_height, img_width).to(trainid_img.device)
    trainid_img = trainid_img.long()
    for train_id_no in range(len(label_to_color)):
        img_colour[:,0,:,:][torch.nonzero(trainid_img==train_id_no, as_tuple=True)] = label_to_color[train_id_no][0]
        img_colour[:,1,:,:][torch.nonzero(trainid_img==train_id_no, as_tuple=True)] = label_to_color[train_id_no][1]
        img_colour[:,2,:,:][torch.nonzero(trainid_img==train_id_no, as_tuple=True)] = label_to_color[train_id_no][2]
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