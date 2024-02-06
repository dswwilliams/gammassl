import sys
import torch
import argparse
import os
import socket
####################################################################################
# custom imports
sys.path.append("../")
####################################################################################

torch.set_printoptions(sci_mode=False, linewidth=200)

########################################################################
## PARSING COMMAND LINE ARGUMENTS ##
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser()
### ###

"""
Notes on arguments:
- the key training options are:
  - model_arch
  - use_proto_seg
  - frozen_target
  - mask_input

TODO: this could possible be improved by consolidating some of the arguments

"""


# ======================== Training ========================
parser.add_argument('--num_train_steps', type=int, default=160000, help="total number of training iterations")
parser.add_argument('--batch_size', type=int, default=24, help="batch size for training (and validation if val_batch_size is None)")
parser.add_argument('--use_proto_seg', type=str2bool, default=False, help="whether to use prototype segmentation")
parser.add_argument('--frozen_target', type=str2bool, default=False,)
parser.add_argument('--model_arch', type=str, default="vit_m2f", help="model architecture: vit_m2f or deeplab")
parser.add_argument('--use_deep_features', type=str2bool, default=True, help="where to extract features from")
parser.add_argument('--train_encoder', type=str2bool, default=False, help="if False, no grads w.r.t. the encoder")
parser.add_argument('--sup_loss_only', type=str2bool, default=False, help="if True, only train with supervised loss")
parser.add_argument("--no_filtering", type=str2bool, default=False, help="method ablation: compute loss over all unlabelled pixels")

# ======================== Model ========================
parser.add_argument('--temperature', type=float, default=0.07, help="temperature for output softmax")
parser.add_argument('--sharpen_temp', type=float, default=None, help="temperature for sharpening of output distribution")
parser.add_argument('--nheads', type=int, default=4, help="number of attention heads for ViT encoder")
parser.add_argument('--vit_size', type=str, default="small", help="size of ViT encoder: small or base")
parser.add_argument('--lora_rank', type=int, default=None, help="if not None, use lora with this rank")
parser.add_argument('--prototype_len', type=int, default=256, help="length of prototype features")
parser.add_argument('--intermediate_dim', type=int, default=256, help="length of decoded features")
parser.add_argument('--use_imagenet_norm', type=str2bool, default=True, help="whether to use imagenet mean and std for normalisation")
parser.add_argument('--gamma_scaling', type=str, default="softmax", help="determines whether gamma is calculated for logits or softmax scores")
parser.add_argument('--gamma_temp', type=float, default=0.1, help="if gamma_scaling is softmax, then this is the temperature used")


# ======================== Loss Calculation and Optimisation ========================
parser.add_argument('--train_vit', type=str2bool, default=True, help="if False, no grads w.r.t. the ViT encoder")
parser.add_argument("--model_weight_decay", type=float, default=0, help="network weight decay")
parser.add_argument("--include_void", type=str2bool, default=False, help="whether include void class in supervised loss")
parser.add_argument("--uniformity_kernel_size", type=int, default=4, help="hyperparam for uniformity loss")
parser.add_argument("--uniformity_stride", type=int, default=4, help="hyperparam for uniformity loss")
parser.add_argument('--num_points', type=int, default=12544, help="hyperparam for mask2former losses")
parser.add_argument('--loss_c_temp', type=float, default=0.1, help="temperature applied for calculating consistency loss")


# ======================== Loss Weighting and Learning Rates ========================
parser.add_argument('--w_c', type=float, default=1, help="weighting for consistency loss")
parser.add_argument('--w_s', type=float, default=1, help="weighting for supervised loss")
parser.add_argument('--w_u', type=float, default=1, help="weighting for uniformity loss")
parser.add_argument('--w_p', type=float, default=1, help="weighting for prototype loss")
parser.add_argument('--w_dice', type=float, default=1, help="weighting for dice segmentation loss")
parser.add_argument('--w_ce', type=float, default=1, help="weighting for mask2former cross entropy loss")
parser.add_argument('--lr', type=float, default=5e-4, help="learning rate for network")
parser.add_argument('--lr_encoder', type=float, default=None, help="learning rate for encoder if not None, else use lr")
parser.add_argument('--lr_policy', type=str, default=None, help="None, poly (decreasing), warmup_poly (warmup then decreasing)")
parser.add_argument('--warmup_ratio', type=float, default=None, help="warmup ratio for learning rate")
parser.add_argument('--n_warmup_iters', type=int, default=None, help="number of warmup iterations")


# ======================== Data Loader and Device  ========================
parser.add_argument('--use_cpu', type=str2bool, default=False, help="override and use cpu")
parser.add_argument('--gpu_no', type=str, default="0", help="which gpu to use")
parser.add_argument('--num_workers', type=int, default=3, help="number of workers for dataloader")


# ======================== Dataset Paths ========================
parser.add_argument('--cityscapes_dataroot', type=str, default="/Volumes/mrgdatastore6/ThirdPartyData/", help="path to cityscapes training dataset")
parser.add_argument('--unlabelled_dataroot', type=str, default="/Volumes/scratchdata/dw/sax_raw", help="path to unlabelled training dataset")
parser.add_argument('--wilddash_dataroot', type=str, default="/Volumes/scratchdata/dw/wilddash/", help="path to wilddash test dataset")
parser.add_argument('--bdd_val_dataroot', type=str, default="/Users/dw/data/bdd_10k", help="path to bdd100k validation dataset")


# ======================== Data Augmentation ========================
parser.add_argument('--mask_input', type=str2bool, default=False, help="data augmentation: whether to mask input images")
parser.add_argument('--no_transforms', type=str2bool, default=False, help="turn off using colour-space transforms with True")
parser.add_argument('--min_crop_ratio', type=float, default=2, help="hyperparam for random crop augmentation")
parser.add_argument('--max_crop_ratio', type=float, default=3, help="hyperparam for random crop augmentation")
parser.add_argument('--random_mask_prob', type=float, default=0.5, help="hyperparam for masking data augmentation")
parser.add_argument("--use_resize_noise", type=str2bool, default=True, help="data augmentation: whether to add noise to resizing")
parser.add_argument('--no_colour', type=str2bool, default=False, help="if True, no colour transforms are applied to training images")


# ======================== Validation Options ========================
# TODO
parser.add_argument('--output_rank_metrics', type=str2bool, default=False, help="normalise uncertainty metric by rank")
# TODO
parser.add_argument('--val_transforms', type=str2bool, default=False, help="whether to colour-transform val images")
parser.add_argument('--val_batch_size', type=int, default=None)
parser.add_argument('--val_every', type=int, default=500, help="frequency of validation w.r.t. number of training iterations")
parser.add_argument('--skip_validation', type=str2bool, default=False, help="whether to skip validation during training")
parser.add_argument('--n_train_segs', type=int, default=4, help="number of qualitative validations viewed from training dataset")
parser.add_argument('--n_val_segs', type=int, default=4, help="number of qualitative validations viewed from val dataset")
parser.add_argument("--max_uncertainty", type=float, default=1, help="upperbound for uncertainty thresholds")
parser.add_argument("--threshold_type", type=str, default="linear", help="how thresholds are distributed: linear or log")
parser.add_argument("--num_thresholds", type=int, default=500, help="number of thresholds used in validation")


# ======================== Logging, Loading and Saving ========================
parser.add_argument('--use_wandb', type=str2bool, default=True, help="whether to use wandb for logging, else use visdom")
parser.add_argument('--detailed_metrics', type=str2bool, default=True, help="whether to log all training metrics, or just key metrics")
parser.add_argument('--log_every', type=int, default=1, help="frequency of logging w.r.t. number of training iterations")
parser.add_argument('--wandb_project', type=str, default="test", help="name of wandb project")
parser.add_argument('--save_every', type=int,  default=1, help="frequency of saving w.r.t. number of times validated")
parser.add_argument('--network_destination', type=str, default=None, help="path to save network")
parser.add_argument('--save_path', type=str,  default=None, help="path from which to load saved model")
parser.add_argument('--prototypes_path', type=str,  default=None, help="path from which to load saved prototypes")
parser.add_argument('--dino_path', type=str, default="/Users/dw/code/pytorch/gammassl/models/dinov2.pth", help="path to dino model weights")
parser.add_argument('--frozen_target_save_path', type=str, default=None)


opt = parser.parse_args()
if socket.gethostname() == "smaug":
    opt.cityscapes_dataroot = "/home/dsww/data/"
    opt.unlabelled_dataroot = "/mnt/data/bdd100k"
    opt.dino_path = "/home/dsww/networks/dinov2/dinov2.pth"
    opt.bdd_val_dataroot = "/home/dsww/data/bdd_10k"
elif opt.use_cpu:
    if "eng" in socket.gethostname():
      opt.batch_size = 2
      opt.num_workers = 0
      opt.cityscapes_dataroot = "/Users/dw/data/"
      opt.unlabelled_dataroot = "/Users/dw/data/bdd100k"
      opt.wilddash_dataroot = "/Users/dw/data/wilddash"
      opt.sax_raw_dataroot = "/Users/dw/data/sax_raw"
      opt.sax_labelled_dataroot = "/Users/dw/data/sax_labelled"
      opt.sensor_models_path = "/Users/dw/code/lut/sensor-models"
    else:
       opt.dino_path = "/Users/dw/networks/dinov2.pth"
       
# print(opt)
########################################################################


torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    from gammassl_trainer import Trainer
    trainer = Trainer(opt)
    trainer.train()