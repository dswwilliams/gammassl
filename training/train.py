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


### ###
parser.add_argument('--gamma_reject_prop', type=float, default=0.5)
### objective opts ###
parser.add_argument('--mask_input', type=str2bool, default=False)
parser.add_argument('--use_proto_seg', type=str2bool, default=False)
parser.add_argument('--train_encoder', type=str2bool, default=False)
parser.add_argument('--model_arch', type=str, default="vit_m2f")




parser.add_argument('--no_colour', type=str2bool, default=False)
parser.add_argument('--use_sax_png_dataset', type=str2bool, default=False)

parser.add_argument('--bdd_raw_dataroot', type=str, default="/Users/dw/data/bdd100k")
parser.add_argument('--train_on_bdd', type=str2bool, default=False)
parser.add_argument('--val_on_bdd', type=str2bool, default=False)
parser.add_argument('--bdd_val_dataroot', type=str, default="/Users/dw/data/bdd_10k")
parser.add_argument('--warmup_masking_num_its', type=int, default=0)
parser.add_argument('--loss_c_masked_only', type=str2bool, default=False)
parser.add_argument('--use_dinov1', type=str2bool, default=False)
parser.add_argument('--dinov1_repo_path', type=str, default="/Users/dw/code/pytorch/dino")
parser.add_argument('--dinov1_path', type=str, default="/Users/dw/networks/dinov1.pth")
parser.add_argument('--ms_imgs_from_unmasked', type=str2bool, default=True)
parser.add_argument('--mlp_dropout_prob', type=float, default=0.0)
parser.add_argument('--attn_dropout_prob', type=float, default=0.0)
parser.add_argument('--run_sup_task_only', type=str2bool, default=False)
parser.add_argument('--no_random_seed', type=str2bool, default=False)
parser.add_argument('--semihard_unmasked_prop', type=float, default=0.5)
parser.add_argument('--use_semihard', type=str2bool, default=False)
parser.add_argument('--masking_model_save_path', type=str, default=None)
parser.add_argument('--use_masking_model', type=str2bool, default=False)
parser.add_argument('--use_soft_mask', type=str2bool, default=False)
parser.add_argument('--use_gumbel_aug', type=str2bool, default=False)
parser.add_argument('--mask_temp', type=float, default=1)
parser.add_argument('--use_topk_aug', type=str2bool, default=False)
parser.add_argument('--topk_prop', type=float, default=0.5)
parser.add_argument('--use_topk', type=str2bool, default=False)
parser.add_argument('--gumbel_temp', type=float, default=1)
parser.add_argument('--use_gumbel', type=str2bool, default=False)
parser.add_argument('--masking_net', type=str, default="cnn")
parser.add_argument('--w_masking_prob', type=float, default=1)
parser.add_argument('--learned_masking_prob', type=float, default=0.5)
parser.add_argument('--lr_masking', type=float, default=1e-4)
parser.add_argument('--run_mask_learning_task', type=str2bool, default=False)
parser.add_argument('--cu_loss', type=str, default="xent")
parser.add_argument('--w_fu', type=float, default=1)
parser.add_argument('--w_c_cu', type=float, default=1)
parser.add_argument('--w_c_ic', type=float, default=1)
parser.add_argument('--optim_middle', type=str2bool, default=False)
parser.add_argument('--mask_prob_total_iters', type=int, default=125)
parser.add_argument('--get_new_mask_prob_every', type=int, default=125)
parser.add_argument('--mask_prob_schedule', type=str, default="random")
parser.add_argument('--min_mask_prob', type=float, default=None)
parser.add_argument('--max_mask_prob', type=float, default=None)
parser.add_argument('--use_sigmoid_for_query_unmasked', type=str2bool, default=True)
parser.add_argument('--soft_consistency_fn', type=str, default="xent")
parser.add_argument('--val_soft_consistency', type=str2bool, default=False)
parser.add_argument('--mask_both', type=str2bool, default=False)
parser.add_argument('--val_transforms', type=str2bool, default=False)
parser.add_argument('--query_mask_region', type=str, default="uncertain")
parser.add_argument('--val_only_consistency', type=str2bool, default=False)
parser.add_argument('--output_rank_metrics', type=str2bool, default=False)
parser.add_argument('--val_with_sigmoid', type=str2bool, default=True)
parser.add_argument('--val_temp', type=float, default=1)
parser.add_argument('--query_mask_temp', type=float, default=1)
parser.add_argument('--query_mask_scaling', type=str, default=None)
parser.add_argument('--kl_temp', type=float, default=1)
parser.add_argument('--loss_c_temp', type=float, default=1)
parser.add_argument('--gamma_temp', type=float, default=1)
parser.add_argument('--gamma_scaling', type=str, default=None)
parser.add_argument('--no_mask_grads', type=str2bool, default=False)
parser.add_argument('--mask_threshold', type=float, default=0.9)
parser.add_argument('--use_sigmoid', type=str2bool, default=True)


parser.add_argument('--wandb_project', type=str, default="test")
parser.add_argument('--frozen_target_save_path', type=str, default=None)
parser.add_argument('--frozen_target', type=str2bool, default=False)






parser.add_argument('--sharpen_temp', type=float, default=None, help="temperature for sharpening of output distribution")

parser.add_argument('--nheads', type=int, default=4, help="number of attention heads for ViT encoder")
parser.add_argument('--num_points', type=int, default=12544, help="hyperparam for mask2former losses")


parser.add_argument('--no_unlabelled', type=str2bool, default=False, help="if True, no unlabelled data is output from dataloader")

parser.add_argument('--vit_size', type=str, default="small", help="size of ViT encoder: small or base")
parser.add_argument('--n_train_segs', type=int, default=4, help="number of qualitative validations viewed from training dataset")
parser.add_argument('--n_val_segs', type=int, default=4, help="number of qualitative validations viewed from val dataset")
parser.add_argument('--train_vit', type=str2bool, default=True, help="if False, no grads w.r.t. the ViT encoder")
parser.add_argument('--lora_rank', type=int, default=None, help="if not None, use lora with this rank")


parser.add_argument('--dino_repo_path', type=str, default="/Users/dw/code/pytorch/dinov2", help="path to dino repo")
parser.add_argument('--dino_path', type=str, default="/Users/dw/code/pytorch/gammassl/models/dinov2.pth", help="path to dino model weights")


parser.add_argument("--model_weight_decay", type=float, default=0, help="network weight decay")
parser.add_argument("--no_filtering", type=str2bool, default=False, help="method ablation: compute loss over all unlabelled pixels")
parser.add_argument("--use_resize_noise", type=str2bool, default=True, help="data augmentation: whether to add noise to resizing")
parser.add_argument("--uniformity_kernel_size", type=int, default=4, help="hyperparam for uniformity loss")
parser.add_argument("--uniformity_stride", type=int, default=4, help="hyperparam for uniformity loss")
parser.add_argument("--include_void", type=str2bool, default=False, help="whether include void class in supervised loss")
### validation ###
parser.add_argument("--max_uncertainty", type=float, default=1, help="upperbound for uncertainty thresholds")
parser.add_argument("--threshold_type", type=str, default="linear", help="how thresholds are distributed: linear or log")
parser.add_argument("--num_thresholds", type=int, default=500, help="number of thresholds used in validation")


### learning rates ###
parser.add_argument('--lr', type=float, default=5e-4, help="learning rate for network")
parser.add_argument('--lr_encoder', type=float, default=None, help="learning rate for encoder if not None, else use lr")
parser.add_argument('--lr_policy', type=str, default=None, help="None, poly (decreasing), warmup_poly (warmup then decreasing)")
parser.add_argument('--warmup_ratio', type=float, default=None, help="warmup ratio for learning rate")
parser.add_argument('--n_warmup_iters', type=int, default=None, help="number of warmup iterations")
parser.add_argument('--total_iters', type=int, default=160000, help="total number of training iterations")
### loss weighting ###
parser.add_argument('--w_c', type=float, default=1, help="weighting for consistency loss")
parser.add_argument('--w_s', type=float, default=1, help="weighting for supervised loss")
parser.add_argument('--w_u', type=float, default=1, help="weighting for uniformity loss")
parser.add_argument('--w_p', type=float, default=1, help="weighting for prototype loss")
parser.add_argument('--w_dice', type=float, default=1, help="weighting for dice segmentation loss")
parser.add_argument('--w_ce', type=float, default=1, help="weighting for mask2former cross entropy loss")
### ###
parser.add_argument('--batch_size', type=int, default=24, help="batch size for training (and validation if val_batch_size is None)")
parser.add_argument('--val_batch_size', type=int, default=None)
parser.add_argument('--no_transforms', type=str2bool, default=False, help="turn off using colour-space transforms with True")
parser.add_argument('--prototype_len', type=int, default=256, help="length of prototype features")
parser.add_argument('--intermediate_dim', type=int, default=256, help="length of decoded features")

### data opts ###
parser.add_argument('--cityscapes_dataroot', type=str, default="/Volumes/mrgdatastore6/ThirdPartyData/", help="path to cityscapes training dataset")
parser.add_argument('--unlabelled_dataroot', type=str, default="/Volumes/scratchdata/dw/sax_raw", help="path to unlabelled training dataset")
parser.add_argument('--wilddash_dataroot', type=str, default="/Volumes/scratchdata/dw/wilddash/", help="path to wilddash test dataset")
parser.add_argument('--min_crop_ratio', type=float, default=2, help="hyperparam for random crop augmentation")
parser.add_argument('--max_crop_ratio', type=float, default=3, help="hyperparam for random crop augmentation")
parser.add_argument('--random_mask_prob', type=float, default=None, help="hyperparam for masking data augmentation")
### device opts ###
parser.add_argument('--use_cpu', type=str2bool, default=False, help="override and use cpu")
parser.add_argument('--gpu_no', type=str, default="0", help="which gpu to use")
parser.add_argument('--num_workers', type=int, default=3, help="number of workers for dataloader")
### network opts ###
parser.add_argument('--use_deep_features', type=str2bool, default=True, help="where to extract features from")
parser.add_argument('--use_imagenet_norm', type=str2bool, default=True, help="whether to use imagenet mean and std for normalisation")
parser.add_argument('--temperature', type=float, default=0.07, help="temperature for output softmax")
### training log opts ###
parser.add_argument('--log_every', type=int, default=1, help="frequency of logging w.r.t. number of training iterations")
### validation opts ###
parser.add_argument('--val_every', type=int, default=500, help="frequency of validation w.r.t. number of training iterations")
parser.add_argument('--skip_validation', type=str2bool, default=False, help="whether to skip validation during training")
### saving opts ###
parser.add_argument('--save_every', type=int,  default=1, help="frequency of saving w.r.t. number of times validated")
parser.add_argument('--network_destination', type=str, default=None, help="path to save network")
parser.add_argument('--save_path', type=str,  default=None, help="path from which to load saved model")
parser.add_argument('--prototypes_path', type=str,  default=None, help="path from which to load saved prototypes")


opt = parser.parse_args()
if socket.gethostname() == "smaug":
    opt.cityscapes_dataroot = "/home/dsww/data/"
    opt.unlabelled_dataroot = "/mnt/data/bdd100k"
    opt.dino_path = "/home/dsww/networks/dinov2/dinov2.pth"
    opt.dino_repo_path = "/home/dsww/code/dinov2"
    opt.bdd_val_dataroot = "/home/dsww/data/bdd_10k"
elif opt.use_cpu:
    opt.batch_size = 2
    opt.num_workers = 0
    opt.cityscapes_dataroot = "/Users/dw/data/"
    opt.unlabelled_dataroot = "/Users/dw/data/bdd100k"
    opt.wilddash_dataroot = "/Users/dw/data/wilddash"
    opt.sax_raw_dataroot = "/Users/dw/data/sax_raw"
    opt.sax_labelled_dataroot = "/Users/dw/data/sax_labelled"
    opt.sensor_models_path = "/Users/dw/code/lut/sensor-models"
# print(opt)
########################################################################


### get trainer ###
torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    from gammassl_trainer import Trainer
    trainer = Trainer(opt)
    trainer.train()