import torch
import torch.nn.functional as F
from utils.device_utils import to_device
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils.test_metric_utils import calculate_miou

def get_thresholds(y_score, num_thresholds):
    """ calculating thresholds based on distribution of logits """
    bin_counts, bin_edges = torch.histogram(y_score, bins=20)
    thresholds_per_bin = torch.round((bin_counts / bin_counts.sum()) * num_thresholds).long()
    thresholds = torch.cat([torch.linspace(bin_edges[i], bin_edges[i+1], steps=thresholds_per_bin[i].item()) for i in range(bin_edges.shape[0]-1)])
    thresholds = torch.unique(thresholds).sort()[0]  # removing repeated values
    return thresholds


def get_nansum_and_count(a, dim):
    """
    - we want to sum across dim, whilst ignoring NaNs
    - however if EVERY element across this dim is NaN, we want to return NaN and not 0 (as pytorch nansum would return)
    """
    b = a.nansum(dim)
    not_nan_count = (~a.isnan()).sum(dim)
    if (not_nan_count == 0).any():
        try:
            b[torch.where(not_nan_count == 0)] = float("nan")
        except:
            pass
    return b, not_nan_count

def tensor_nansum(a, b):
    """ takes two tensors and computes nansum of them """
    assert len(a.shape) == len(b.shape)
    new_dim = len(a.shape)
    sum = get_nansum_and_count(torch.cat((a.unsqueeze(new_dim), b.unsqueeze(new_dim)), dim=new_dim), dim=new_dim)[0]
    return sum


def update_running_variable(running_variable_total, running_variable_count, update, batch_dim):
    # aggregate metric across batch dim (NaNs do not contribute to sum or count)
    batch_aggregated_update, count = get_nansum_and_count(update, dim=batch_dim)
    # combine running variable and update
    if len(batch_aggregated_update.shape) == 0:
        batch_aggregated_update = batch_aggregated_update.unsqueeze(0)
    running_variable_total = tensor_nansum(running_variable_total, batch_aggregated_update)
    running_variable_count += count
    del count, update
    return running_variable_total, running_variable_count


@torch.no_grad()
def perform_batch_ue_validation(
                        val_imgs, 
                        val_labels,
                        model,
                        opt,
                        global_thresholds,
                        writer,
                        branch,
                        ):
    ### segment imgs ###
    if branch == "query":
        if opt.use_symmetric_branches:
            # query branch doesn't use prototypes
            seg_masks_K = model.seg_net.get_query_seg_masks(val_imgs, include_void=False, high_res=True)
        else:
            # query branch uses prototypes
            seg_masks_K = model.proto_segment_imgs(val_imgs, use_dataset_prototypes=True)
        seg_masks_K = torch.softmax(seg_masks_K/opt.temperature, dim=1)
        seg_masks_K = F.interpolate(seg_masks_K, size=(val_imgs.shape[2], val_imgs.shape[3]), mode="bilinear", align_corners=True)
        ms_imgs, segs_K = torch.max(seg_masks_K, dim=1)
        uncertainty_maps = 1 - ms_imgs
    elif branch == "target":
        # target doesn't use prototypes regardless of opt.use_symmetric_branches
        if opt.frozen_target:
            seg_masks_K = model.target_seg_net.get_target_seg_masks(val_imgs, include_void=False, high_res=True)
        else:
            seg_masks_K = model.seg_net.get_target_seg_masks(val_imgs, include_void=False, high_res=True)
        seg_masks_K = torch.softmax(seg_masks_K/opt.temperature, dim=1)
        ms_imgs, segs_K = torch.max(seg_masks_K, dim=1)
        uncertainty_maps = 1 - ms_imgs

    # writer.add_histogram("uncertainty_maps", uncertainty_maps, 0)

    ### varying p(certain) and calculating certainty metrics ###
    ue_metrics, local_thresholds = calculate_ue_metrics(
                                segmentations=segs_K,
                                labels=val_labels,
                                uncertainty_maps=uncertainty_maps,
                                max_uncertainty=opt.max_uncertainty,
                                num_thresholds=opt.num_thresholds,
                                threshold_type=opt.threshold_type,
                                )
    
    ue_metrics["miou"] = calculate_miou(segmentations=segs_K, labels=val_labels, num_classes=model.num_known_classes+1)
    return ue_metrics, local_thresholds

@torch.no_grad()
def perform_batch_seg_validation(
                        val_imgs, 
                        val_labels,
                        model,
                        opt,
                        use_seg_head=True,
                        ):
    ### segment imgs ###
    if use_seg_head:
        seg_masks_K = model.seg_net.get_seg_masks(val_imgs, include_void=False, high_res=True)
        seg_masks_K = torch.softmax(seg_masks_K, dim=1)
        segs_K = torch.argmax(seg_masks_K, dim=1)
    else:
        seg_masks_K = model.proto_segment_imgs(val_imgs, use_dataset_prototypes=True)
        seg_masks_K = F.interpolate(seg_masks_K, size=(val_imgs.shape[2], val_imgs.shape[3]), mode="bilinear", align_corners=True)
        seg_masks_K = torch.softmax(seg_masks_K/opt.temperature, dim=1)
        segs_K = torch.argmax(seg_masks_K, dim=1)


    ### calculating seg metrics ###
    seg_metrics = calculate_seg_metrics(
                                segmentations=segs_K,
                                labels=val_labels,
                                )
    return seg_metrics


def init_val_ue_metrics(num_thresholds, counts=False):
    metric_names = ["n_inaccurate_and_certain", "n_accurate_and_certain", "n_uncertain_and_accurate", "n_uncertain_and_inaccurate", "miou"]

    ue_metrics_totals = {name: torch.zeros(num_thresholds) for name in metric_names}
    if counts:
        ue_metrics_counts = {name: torch.zeros(num_thresholds) for name in metric_names}
        return ue_metrics_totals, ue_metrics_counts
    else:
        return ue_metrics_totals

def init_val_seg_metrics(counts=False):
    metric_names = ["n_inaccurate", "n_accurate, miou"]

    seg_metrics_totals = {name: 0 for name in metric_names}
    if counts:
        seg_metrics_counts = {name: 0 for name in metric_names}
        return seg_metrics_totals, seg_metrics_counts
    else:
        return seg_metrics_totals


def calculate_fbeta_score(tp, fp, fn, beta):
    """
    F1 score = 2*tp /(2*tp + fp + fn)
    """
    fbeta_score = (1+beta**2)*tp /((1+beta**2)*tp + fp + fn)
    return fbeta_score

def calculate_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

@torch.no_grad()
def calculate_ue_metrics(
                    segmentations, 
                    labels, 
                    num_thresholds=50, 
                    uncertainty_maps=None, 
                    max_uncertainty=1, 
                    threshold_type="linear", 
                    ):
    """
    - loop over thresholds
    - for each threshold, confidence_masks changes
    """
    bs = segmentations.shape[0]
    device = segmentations.device

    ################################################################################################################
    ### defining thresholds ###
    if threshold_type == "log":
        thresholds = max_uncertainty * torch.logspace(start=-15, end=0, steps=num_thresholds, base=2)        # range: [0, max_uncertainty]
    elif threshold_type == "linear":
        thresholds = max_uncertainty * torch.linspace(0, 1, steps=num_thresholds)                            # range: [0, max_uncertainty]
    elif threshold_type == "scaled":
        thresholds = get_thresholds(uncertainty_maps, num_thresholds)       # NB: thresholds will be different for each batch, also will return different len
    ################################################################################################################

    ################################################################################################################
    ### init running variables ###
    val_metrics = {}
    val_metrics["n_uncertain_and_accurate"] = to_device(torch.zeros(bs, num_thresholds), device)
    val_metrics["n_uncertain_and_inaccurate"] = to_device(torch.zeros(bs, num_thresholds), device)
    val_metrics["n_inaccurate_and_certain"] = to_device(torch.zeros(bs, num_thresholds), device)
    val_metrics["n_accurate_and_certain"] = to_device(torch.zeros(bs, num_thresholds), device)
    ################################################################################################################

    accuracy_masks = torch.eq(segmentations, labels).float()        # where segmentations == labels, 1, else 0
    # loop over threshold values
    for threshold_no in range(thresholds.shape[0]):
        ################################################################################################################
        ### getting confidence_masks ###
        threshold = thresholds[threshold_no]
        confidence_masks = torch.le(uncertainty_maps, threshold).float()           # 1 if uncertainty_maps <= threshold, else 0
        ################################################################################################################
        
        ################################################################################################################
        ### calculating uncertainty estimation metrics ###
        n_accurate_and_certain = (accuracy_masks * confidence_masks).sum((1,2))
        n_inaccurate_and_certain = ((1-accuracy_masks) * confidence_masks).sum((1,2))
        n_uncertain_and_accurate = (accuracy_masks * (1-confidence_masks)).sum((1,2))
        n_uncertain_and_inaccurate = ((1-accuracy_masks) * (1-confidence_masks)).sum((1,2))
        
        val_metrics["n_inaccurate_and_certain"][:, threshold_no] = n_inaccurate_and_certain
        val_metrics["n_accurate_and_certain"][:, threshold_no] = n_accurate_and_certain
        val_metrics["n_uncertain_and_accurate"][:, threshold_no] = n_uncertain_and_accurate
        val_metrics["n_uncertain_and_inaccurate"][:, threshold_no] = n_uncertain_and_inaccurate
        ################################################################################################################
    return val_metrics, thresholds

@torch.no_grad()
def calculate_seg_metrics(
                    segmentations, 
                    labels, 
                    ):
    """
    - loop over thresholds
    - for each threshold, confidence_masks changes
    """
    bs = segmentations.shape[0]
    device = segmentations.device

    ################################################################################################################
    ### init running variables ###
    val_metrics = {}
    val_metrics["n_accurate"] = to_device(torch.zeros(bs), device)
    val_metrics["n_inaccurate"] = to_device(torch.zeros(bs), device)
    ################################################################################################################

    accuracy_masks = torch.eq(segmentations, labels).float()        # where segmentations == labels, 1, else 0

    val_metrics["n_accurate"] = accuracy_masks.sum((1,2))
    val_metrics["n_inaccurate"] = (1-accuracy_masks).sum((1,2))
    return val_metrics


def plot_val_ue_metrics_to_tensorboard(val_metrics, writer, validation_count, dataset_name):
    # the scalars for the x-axis in tensorboard have to ascend
    if val_metrics["p_certain"][:len(val_metrics["p_certain"])//2].mean() > val_metrics["p_certain"][len(val_metrics["p_certain"])//2:].mean():
        val_metrics["p_certain"] = val_metrics["p_certain"].flip(0)
        val_metrics["p_accurate"] = val_metrics["p_accurate"].flip(0)
        val_metrics["fhalf"] = val_metrics["fhalf"].flip(0)
        val_metrics["acc_md"] = val_metrics["acc_md"].flip(0)

    if val_metrics["recall"][:len(val_metrics["recall"])//2].mean() > val_metrics["recall"][len(val_metrics["recall"])//2:].mean():
        val_metrics["recall"] = val_metrics["recall"].flip(0)
        val_metrics["precision"] = val_metrics["precision"].flip(0)
        
    # create a matplotlib figure instead of using tensorboard
    fig, ax = plt.subplots()
    ax.plot(val_metrics["p_certain"].cpu().numpy(), val_metrics["acc_md"].cpu().numpy())
    ax.set_xlabel("p_certain")
    ax.set_ylabel("A_md")
    wandb.log({"A_md "+str(dataset_name)+'/'+str(validation_count): fig}, commit=False)


    fig, ax = plt.subplots() 
    ax.plot(val_metrics["p_certain"].cpu().numpy(), val_metrics["fhalf"].cpu().numpy())
    ax.set_xlabel("p_certain")
    ax.set_ylabel("F_0.5")
    wandb.log({"F_0.5 "+str(dataset_name)+'/'+str(validation_count): fig}, commit=False)

    fig, ax = plt.subplots()
    ax.plot(val_metrics["recall"].cpu().numpy(), val_metrics["precision"].cpu().numpy())
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    wandb.log({"Precision vs Recall "+str(dataset_name)+'/'+str(validation_count): fig}, commit=False)


    # plotting aggregated metrics (i.e. single datum per validation step)
    wandb.log({"Validation: "+str(dataset_name)+"/A_md": val_metrics["acc_md"].max().float().item()}, commit=False)
    wandb.log({"Validation: "+str(dataset_name)+"/F_0.5": val_metrics["fhalf"].max().float().item()}, commit=False)
    wandb.log({"Validation: "+str(dataset_name)+"/Segmentation Accuracy": val_metrics["p_accurate"][0].float().item()}, commit=False)
    wandb.log({"Validation: "+str(dataset_name)+"/Mean IoU": val_metrics["miou"][0].float().item()}, commit=False)


def plot_val_seg_metrics_to_tensorboard(val_metrics, writer, validation_count, dataset_name):
    # writer.add_scalar('Validation: '+str(dataset_name)+'/Segmentation Accuracy (SegHead)', val_metrics["p_accurate"].float().item(), int(validation_count))
    wandb.log({"Validation: "+str(dataset_name)+"/Segmentation Accuracy (SegHead)": val_metrics["p_accurate"].float().item()}, commit=False)

