import torch

@torch.no_grad()
def calculate_ue_metrics(
                    segmentations, 
                    labels, 
                    num_thresholds=100, 
                    uncertainty_maps=None, 
                    max_uncertainty=1, 
                    threshold_type="linear", 
                    ):

    bs = segmentations.shape[0]
    device = segmentations.device

    ################################################################################################################
    ### defining thresholds ###
    if threshold_type == "log":
        thresholds = max_uncertainty * torch.logspace(start=-15, end=0, steps=num_thresholds, base=2)        # range: [0, max_uncertainty]
    elif threshold_type == "linear":
        thresholds = max_uncertainty * torch.linspace(0, 1, steps=num_thresholds)                            # range: [0, max_uncertainty]
    ################################################################################################################

    ################################################################################################################
    ### init running variables ###
    val_metrics = {}
    val_metrics["n_uncertain_and_accurate"] = torch.zeros(bs, num_thresholds)
    val_metrics["n_uncertain_and_inaccurate"] = torch.zeros(bs, num_thresholds)
    val_metrics["n_inaccurate_and_certain"] = torch.zeros(bs, num_thresholds)
    val_metrics["n_accurate_and_certain"] = torch.zeros(bs, num_thresholds)
    ################################################################################################################

    accuracy_masks = torch.eq(segmentations, labels).float()        # where segmentations == labels, 1, else 0
    inaccuracy_masks = 1 - accuracy_masks                           # where segmentations != labels, 1, else 0
    
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
        n_inaccurate_and_certain = (inaccuracy_masks * confidence_masks).sum((1,2))
        n_uncertain_and_accurate = (accuracy_masks * (1-confidence_masks)).sum((1,2))
        n_uncertain_and_inaccurate = (inaccuracy_masks * (1-confidence_masks)).sum((1,2))
        
        val_metrics["n_inaccurate_and_certain"][:, threshold_no] = n_inaccurate_and_certain
        val_metrics["n_accurate_and_certain"][:, threshold_no] = n_accurate_and_certain
        val_metrics["n_uncertain_and_accurate"][:, threshold_no] = n_uncertain_and_accurate
        val_metrics["n_uncertain_and_inaccurate"][:, threshold_no] = n_uncertain_and_inaccurate
        ################################################################################################################
    return val_metrics

@torch.no_grad()
def calculate_ue_metrics_new(segmentations, labels, num_thresholds=100, uncertainty_maps=None, max_uncertainty=1, threshold_type="linear"):
    bs = segmentations.shape[0]
    device = segmentations.device

    ### defining thresholds ###
    if threshold_type == "log":
        thresholds = max_uncertainty * torch.logspace(start=-15, end=0, steps=num_thresholds, base=2)
    elif threshold_type == "linear":
        thresholds = max_uncertainty * torch.linspace(0, 1, steps=num_thresholds)

    ### init running variables ###
    val_metrics = {metric: torch.zeros(bs, num_thresholds) for metric in ["n_uncertain_and_accurate", "n_uncertain_and_inaccurate", "n_inaccurate_and_certain", "n_accurate_and_certain"]}

    accuracy_masks = torch.eq(segmentations, labels).float().flatten()
    inaccuracy_masks = 1 - accuracy_masks

    # Flatten and sort the uncertainty maps
    sort_idxs = torch.argsort(uncertainty_maps.flatten())
    sorted_uncertainties = uncertainty_maps.flatten()[sort_idxs]
    sorted_accuracies = accuracy_masks[sort_idxs]
    sorted_inaccuracies = inaccuracy_masks[sort_idxs]

    # Compute indices for thresholds
    indices_for_thresholds = [torch.searchsorted(sorted_uncertainties, threshold, right=True) for threshold in thresholds]

    for i, count in enumerate(indices_for_thresholds):
        if count >= sorted_uncertainties.numel():
            count = sorted_uncertainties.numel() - 1

        n_accurate_and_certain = sorted_accuracies[:count].sum()
        n_inaccurate_and_certain = sorted_inaccuracies[:count].sum()
        n_uncertain_and_accurate = sorted_accuracies[count:].sum()
        n_uncertain_and_inaccurate = sorted_inaccuracies[count:].sum()

        val_metrics["n_accurate_and_certain"][:, i] = n_accurate_and_certain
        val_metrics["n_inaccurate_and_certain"][:, i] = n_inaccurate_and_certain
        val_metrics["n_uncertain_and_accurate"][:, i] = n_uncertain_and_accurate
        val_metrics["n_uncertain_and_inaccurate"][:, i] = n_uncertain_and_inaccurate

    return val_metrics


import time

bs = 2
K = 19
h = w = 16
NUM_THRESHOLDS = 1000
seg_masks = torch.randn(bs, K, h, w)
seg_masks = torch.softmax(seg_masks, dim=1) 
labels = torch.randint(low=0, high=K, size=(bs, h, w))

ms_imgs, segmentations = seg_masks.max(dim=1)

start_time = time.time()
for _ in range(100):
    calculate_ue_metrics(segmentations, labels, num_thresholds=NUM_THRESHOLDS, uncertainty_maps=(1-ms_imgs), max_uncertainty=1, threshold_type="linear")
    # calculate_ue_metrics_new(segmentations, labels, num_thresholds=NUM_THRESHOLDS, uncertainty_maps=(1-ms_imgs), max_uncertainty=1, threshold_type="linear")
print(f"Time taken: {time.time() - start_time}")


