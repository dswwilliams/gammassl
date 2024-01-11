import sys
import torch
import argparse
import os
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
parser.add_argument('--val_batch_size', type=int, default=None)
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
parser.add_argument('--adaptive_masking_p', type=float, default=0.5)
parser.add_argument('--use_adaptive_masking', type=str2bool, default=False)
parser.add_argument('--softmax_inference', type=str2bool, default=True)
parser.add_argument('--loss_c_unmasked_only', type=str2bool, default=False)
parser.add_argument('--proj_fu_features', type=str2bool, default=False)
parser.add_argument('--use_feature_uniform_loss', type=str2bool, default=False)
parser.add_argument('--use_m2f_xent', type=str2bool, default=False)
parser.add_argument('--w_kl', type=float, default=1)
parser.add_argument('--use_kl_uniform_loss', type=str2bool, default=False)
parser.add_argument('--mask_only_query', type=str2bool, default=True)
parser.add_argument('--wandb_project', type=str, default="gammassl")
parser.add_argument('--region_proposal_testing', type=str2bool, default=False)
parser.add_argument('--run_sup_masking_training', type=str2bool, default=False)
parser.add_argument('--frozen_target_save_path', type=str, default=None)
parser.add_argument('--frozen_target', type=str2bool, default=False)
parser.add_argument('--proj_query_pup', type=str2bool, default=False)
parser.add_argument('--enc_td_no_query_grad', type=str2bool, default=False)
parser.add_argument('--td_no_query_grad', type=str2bool, default=False)
parser.add_argument('--sharpen_temp', type=float, default=None)
parser.add_argument('--ema_update_every', type=int, default=1)
parser.add_argument('--ema_beta', type=float, default=0.999)
parser.add_argument('--use_ema_target_net', type=str2bool, default=False)
parser.add_argument('--target_temp_factor', type=float, default=0.1)
parser.add_argument('--use_symmetric_branches', type=str2bool, default=False)
parser.add_argument('--validate_only', type=str2bool, default=False)
parser.add_argument('--qual_validate', type=str2bool, default=False)
parser.add_argument('--qual_output_dir', type=str, default=f"{os.path.expanduser('~')}/processed_results/qual_results")
parser.add_argument('--skip_uniformity', type=str2bool, default=False)
parser.add_argument('--skip_projection', type=str2bool, default=False)
parser.add_argument('--m2f_query_test', type=str2bool, default=False)
parser.add_argument('--nheads', type=int, default=4)
parser.add_argument('--num_points', type=int, default=12544)
parser.add_argument('--run_sup_task', type=str2bool, default=False)
parser.add_argument('--random_mask_prob', type=float, default=None)
parser.add_argument('--run_masking_task', type=str2bool, default=False)
parser.add_argument('--use_scannet_twice', type=str2bool, default=False)
parser.add_argument('--proto_seg_for_sup', type=str2bool, default=False)
parser.add_argument('--imagenet_dataroot', type=str, default="")
parser.add_argument('--scannet_dataroot', type=str, default=None)
parser.add_argument('--scannet', type=str2bool, default=False)
parser.add_argument('--weird_dataroot', type=str, default=None)
parser.add_argument('--no_unlabelled', type=str2bool, default=False)
parser.add_argument('--sunrgbd_dataroot', type=str, default=None)
parser.add_argument('--sunrgbd', type=str2bool, default=False)
parser.add_argument('--val_all_sax', type=str2bool, default=False)
parser.add_argument('--vit_size', type=str, default="small")
parser.add_argument('--n_train_segs', type=int, default=4)
parser.add_argument('--n_val_segs', type=int, default=4)
parser.add_argument('--use_wandb', type=str2bool, default=True)
parser.add_argument('--train_vit', type=str2bool, default=True, help="whether to update all vit params")
parser.add_argument('--exp_name', type=str, default=None, help="experiment name")
parser.add_argument('--lora_rank', type=int, default=None)
parser.add_argument('--dino_repo_path', type=str, default="/Users/dw/code/pytorch/dinov2")
parser.add_argument('--dino_path', type=str, default="/Users/dw/code/pytorch/gammassl/models/dinov2.pth")
parser.add_argument('--cityscapes_only', type=str2bool, default=False)
parser.add_argument('--use_vit_adapter', type=str2bool, default=False)
parser.add_argument("--decode_head", type=str, default="setr")
parser.add_argument("--use_dino", type=str2bool, default=False)
parser.add_argument("--use_fake_data", type=str2bool, default=False)
parser.add_argument("--use_fixed_random_init", type=str2bool, default=False)
parser.add_argument("--model_config_path", type=str, default="")
parser.add_argument("--model_weight_decay", type=float, default=0)
parser.add_argument("--no_filtering", type=str2bool, default=False, help="M_gamma to ones")
parser.add_argument("--use_resize_noise", type=str2bool, default=True)
parser.add_argument("--uniformity_kernel_size", type=int, default=4)
parser.add_argument("--uniformity_stride", type=int, default=4)
parser.add_argument("--include_void", type=str2bool, default=False, help="Whether to compute supervised loss on void pixels")
### validation ###
parser.add_argument("--max_uncertainty", type=float, default=1)
parser.add_argument("--threshold_type", type=str, default="linear", help="linear, scaled or log")
parser.add_argument("--num_thresholds", type=int, default=500)
### learning rates ###
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_encoder', type=float, default=None)
parser.add_argument('--warmup_ratio', type=float, default=None)
parser.add_argument('--n_warmup_iters', type=int, default=None)
parser.add_argument('--lr_policy', type=str, default=None)
parser.add_argument('--lr_mult', type=float, default=1)
parser.add_argument('--decay_mult', type=float, default=1)
parser.add_argument('--total_iters', type=int, default=160000)
### loss weighting ###
parser.add_argument('--w_c', type=float, default=1)
parser.add_argument('--w_s', type=float, default=1)
parser.add_argument('--w_gamma', type=float, default=1)
parser.add_argument('--w_u', type=float, default=1)
parser.add_argument('--w_p', type=float, default=1)
parser.add_argument('--w_m', type=float, default=1)
parser.add_argument('--w_mask', type=float, default=1)
parser.add_argument('--w_dice', type=float, default=1)
parser.add_argument('--w_ce', type=float, default=1)
### ###
parser.add_argument('--method', type=str, default="gammassl")
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--use_class_weights_ssl', type=str2bool, default=False)
parser.add_argument('--use_class_weights_sup', type=str2bool, default=False)
parser.add_argument('--no_transforms', type=str2bool, default=False)
parser.add_argument('--overfit_on_batch', type=str2bool, default=False)
parser.add_argument('--feature_len', type=int, default=None)
parser.add_argument('--prototype_len', type=int, default=256)
parser.add_argument('--intermediate_dim', type=int, default=256)

parser.add_argument('--feature_df', type=int, default=4, help="Downsample factor from images to features")
### data opts ###
parser.add_argument('--cityscapes_dataroot', type=str, default="/Volumes/mrgdatastore6/ThirdPartyData/")
parser.add_argument('--wilddash_dataroot', type=str, default="/Volumes/scratchdata/dw/wilddash/")
parser.add_argument('--sax_raw_dataroot', type=str, default="/Volumes/scratchdata/dw/sax_raw")
parser.add_argument('--unlabelled_dataroot', type=str, default="/Volumes/scratchdata/dw/sax_raw")
parser.add_argument('--sax_labelled_dataroot', type=str, default="/Volumes/scratchdata/dw/sax_labelled")
parser.add_argument('--sensor_models_path', type=str, default="/Volumes/scratchdata/dw/lut/sensor-models")
parser.add_argument('--cityscapes_class_weights_path', type=str, default="/Volumes/scratchdata/dw/cityscapes_class_weights_19c.pkl")
parser.add_argument('--benchmarks_results_dir', type=str, default="/Volumes/scratchdata/dw/project_sax_results_tau_eq_1/results")
parser.add_argument('--sax_domain', type=str, default="london")
parser.add_argument('--min_crop_ratio', type=float, default=2)
parser.add_argument('--max_crop_ratio', type=float, default=3)
parser.add_argument('--big_crop_size', type=int, default=208)
### device opts ###
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--use_cpu', type=str2bool, default=False)
parser.add_argument('--gpu_no', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=3)
### network opts ###
parser.add_argument('--use_deep_features', type=str2bool, default=True)
parser.add_argument('--seg_net_config_path', type=str, default=None)
parser.add_argument('--use_imagenet_norm', type=str2bool, default=True)
parser.add_argument('--imagenet_save_path', type=str, default=None)
parser.add_argument('--temperature', type=float, default=0.07)
### training log opts ###
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--log_every', type=int, default=1)
parser.add_argument('--visdom_port', type=int, default=8097)
parser.add_argument('--output2visdom', type=str2bool, default=False)
### validation opts ###
parser.add_argument('--full_validation_every', type=int, default=125)
parser.add_argument('--skip_validation', type=str2bool, default=False)
### saving opts ###
parser.add_argument('--save_every', type=int,  default=1)               # NOTE: defined by number of times validated, not by total number of training its
parser.add_argument('--network_destination', type=str, default=None)
parser.add_argument('--save_path', type=str,  default=None)
parser.add_argument('--prototypes_path', type=str,  default=None)


opt = parser.parse_args()
if opt.use_cpu:
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