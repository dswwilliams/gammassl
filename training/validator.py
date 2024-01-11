import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import cv2
import copy
from utils.validation_utils import init_val_ue_metrics, perform_batch_ue_validation
from utils.validation_utils import plot_val_ue_metrics

from utils.device_utils import to_device

import copy
from gammassl_losses import GammaSSLLosses

def segmasks2gammasegmasks(seg_masks, gamma, opt):
    bs, _, h, w = seg_masks.shape
    device = seg_masks.device
    gammas = gamma * torch.ones(bs, 1, h, w).to(device)      # shape: [bs, 1, h, w]
    gamma_seg_masks = torch.cat((seg_masks, gammas), dim=1)             # shape: [bs, K+1, h, w]
    return gamma_seg_masks


def get_global_metrics_from_local(global_thresholds, local_thresholds, local_metric, global_metric_total, global_metric_count):
    """
    local_thresholds: (num_local_thresholds)
    local_metric: (batch_size, num_local_thresholds)
    global_thresholds: (num_global_thresholds)
    global_metric_total: (num_global_thresholds)
    global_metric_count: (num_global_thresholds)
    """

    local_thresholds = local_thresholds.long()
    global_thresholds = global_thresholds.long()

    masks = (global_thresholds[:, None] == local_thresholds)  # shape: (num_global_thresholds, num_local_thresholds)

    local_metric_expanded = local_metric.unsqueeze(1)  # shape: (batch_size, 1, num_local_thresholds)
    masked_metric = local_metric_expanded * masks  # shape: (batch_size, num_global_thresholds, num_local_thresholds)

    # dont include metrics with nan values in count or total
    global_metric_total += torch.nansum(masked_metric, dim=(0, 2))
    nan_metric_mask = (~local_metric.isnan()).float()
    nan_metric_mask_expanded = nan_metric_mask.unsqueeze(1)
    masked_nan_metric_mask = nan_metric_mask_expanded * masks
    global_metric_count += torch.sum(masked_nan_metric_mask, dim=(0, 2))

    return global_metric_total, global_metric_count

def calculate_mean_stats(n_accurate_and_certain, n_uncertain_and_inaccurate, n_inaccurate_and_certain, n_uncertain_and_accurate):
    total_sum = n_accurate_and_certain + n_uncertain_and_inaccurate + n_inaccurate_and_certain + n_uncertain_and_accurate

    mean_stats = {}
    mean_stats["mean_accurate_and_certain"] = n_accurate_and_certain / total_sum
    mean_stats["mean_uncertain_and_inaccurate"] = n_uncertain_and_inaccurate / total_sum
    mean_stats["mean_inaccurate_and_certain"] = n_inaccurate_and_certain / total_sum
    mean_stats["mean_uncertain_and_accurate"] = n_uncertain_and_accurate / total_sum
    return mean_stats

def calculate_fbeta_score(tp, fp, fn, beta):
    """
    F1 score = 2*tp /(2*tp + fp + fn)
    """
    fbeta_score = (1+beta**2)*tp /((1+beta**2)*tp + fp + (beta**2)*fn)
    return fbeta_score

def calculate_tpr(tp, fn):
    tpr = tp / (tp + fn)
    return tpr

def calculate_fpr(tn, fp):
    fpr = fp / (fp + tn)
    return fpr

def calculate_risk(tp, fp):
    """ 
    - of those pixels seen as certain, how many are inaccurate 
    - same as 1 - precision
    """
    risk = fp / (tp + fp)
    return risk
def calculate_reverse_precision_recall(tp, tn, fp, fn):
    """ 
    - of those pixels seen as certain, how many are inaccurate 
    - same as 1 - precision
    """
    reverse_tp = tn
    reverse_tn = tp
    reverse_fp = fn
    reverse_fn = fp

    reverse_precision = reverse_tp / (reverse_tp + reverse_fp)
    reverse_recall = reverse_tp / (reverse_tp + reverse_fn)
    reverse_precision[reverse_precision.isnan()]
    reverse_precision[torch.nonzero(reverse_precision.isnan())] = 0
    return reverse_precision, reverse_recall

def calculate_precision_recall(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precision[torch.nonzero(precision.isnan())] = 0
    return precision, recall

def calculate_p_accurate_certain(tp, tn, fp, fn):
    p_accurate_certain = tp / (tp + tn + fp + fn)
    return p_accurate_certain

def calculate_p_certain(tp, tn, fp, fn):
    p_certain = (tp+fp) / (tp + tn + fp + fn)
    return p_certain

def calculate_accuracy(tp, tn, fp, fn):
    accuracy = (tp+tn) / (tp + tn + fp + fn)
    return accuracy

def calculate_p_accurate(tp, tn, fp, fn):
    accuracy = (tp+fn) / (tp + tn + fp + fn)
    return accuracy


def calculate_metrics_suite(val_metrics_totals, val_metrics_counts):
    processed_metrics = {}
    
    processed_metrics["precision"], processed_metrics["recall"] = calculate_precision_recall(
                            tp=val_metrics_totals["n_accurate_and_certain"],
                            tn=val_metrics_totals["n_uncertain_and_inaccurate"],
                            fp=val_metrics_totals["n_inaccurate_and_certain"],
                            fn=val_metrics_totals["n_uncertain_and_accurate"],
                            )

    processed_metrics["fhalf"] = calculate_fbeta_score(
                    tp=val_metrics_totals["n_accurate_and_certain"],
                    fp=val_metrics_totals["n_inaccurate_and_certain"],
                    fn=val_metrics_totals["n_uncertain_and_accurate"],
                    beta=0.5,
                    )
    processed_metrics["acc_md"] = calculate_accuracy(
                    tp=val_metrics_totals["n_accurate_and_certain"],
                    tn=val_metrics_totals["n_uncertain_and_inaccurate"],
                    fp=val_metrics_totals["n_inaccurate_and_certain"],
                    fn=val_metrics_totals["n_uncertain_and_accurate"],
                    )

    processed_metrics["p_certain"] = calculate_p_certain(
                    tp=val_metrics_totals["n_accurate_and_certain"],
                    tn=val_metrics_totals["n_uncertain_and_inaccurate"],
                    fp=val_metrics_totals["n_inaccurate_and_certain"],
                    fn=val_metrics_totals["n_uncertain_and_accurate"],
                    )
    processed_metrics["p_accurate"] = calculate_p_accurate(
                    tp=val_metrics_totals["n_accurate_and_certain"],
                    tn=val_metrics_totals["n_uncertain_and_inaccurate"],
                    fp=val_metrics_totals["n_inaccurate_and_certain"],
                    fn=val_metrics_totals["n_uncertain_and_accurate"],
                    )
    
    mean_stats = calculate_mean_stats(
                            n_accurate_and_certain=val_metrics_totals["n_accurate_and_certain"],
                            n_uncertain_and_inaccurate=val_metrics_totals["n_uncertain_and_inaccurate"],
                            n_inaccurate_and_certain=val_metrics_totals["n_inaccurate_and_certain"],
                            n_uncertain_and_accurate=val_metrics_totals["n_uncertain_and_accurate"],
                            )
    processed_metrics["mean_accurate_and_certain"] = mean_stats["mean_accurate_and_certain"]
    processed_metrics["mean_uncertain_and_inaccurate"] = mean_stats["mean_uncertain_and_inaccurate"]
    processed_metrics["mean_inaccurate_and_certain"] = mean_stats["mean_inaccurate_and_certain"]
    processed_metrics["mean_uncertain_and_accurate"] = mean_stats["mean_uncertain_and_accurate"]

    processed_metrics["miou"] = val_metrics_totals["miou"] / val_metrics_counts["miou"]

    return processed_metrics

class Validator():
    def __init__(self, opt, class_labels, writer=None, device=None):
        self.opt = opt
        self.writer = writer

        if device is not None:
            self.device = device

        self.class_labels = copy.deepcopy(class_labels)
        # if the last class is not void (i.e. if just the known classes), add void class
        if not self.class_labels[-1] == "void":
            self.class_labels.append("void")

        # turn class list into class dict, in one line of code
        self.class_dict = {class_idx:class_label  for class_idx, class_label in enumerate(self.class_labels)}

        self.losses = GammaSSLLosses(self.opt, self.device, num_known_classes=len(self.class_labels))

        self.val_seg_idxs = {}

    ######################################################################################################################################################
    @torch.no_grad()
    def view_val_segmentations(self, val_dataset, model):
        device = next(model.parameters()).device

        dataset_name = val_dataset.name

        print("viewing val segmentations for {}".format(dataset_name))

        self.val_seg_idxs[dataset_name] = [int(idx) for idx in self.val_seg_idxs[dataset_name]]
        val_dataset = torch.utils.data.Subset(val_dataset, self.val_seg_idxs[dataset_name])

        if self.opt.mask_input:
            from utils.collation_utils import get_val_collate_fn
            val_collate_fn = get_val_collate_fn(
                                    img_size=val_dataset.crop_sizes, 
                                    patch_size=model.patch_size, 
                                    random_mask_prob=self.opt.random_mask_prob, 
                                    min_mask_prob=self.opt.min_mask_prob, 
                                    max_mask_prob=self.opt.max_mask_prob,
                                    )
        else:
            val_collate_fn = None

        _num_workers = 0 if self.opt.num_workers == 0 else 2

        full_val_dataloader = torch.utils.data.DataLoader(
                                                    dataset=val_dataset, 
                                                    batch_size=min(self.opt.batch_size, len(val_dataset)), 
                                                    shuffle=False, 
                                                    num_workers=_num_workers, 
                                                    drop_last=False,
                                                    collate_fn=val_collate_fn,
                                                    )
        seg_count = 0
        iterator = tqdm(full_val_dataloader)
        for _, (val_dict) in enumerate(iterator):
            val_imgs = to_device(val_dict["img"], device)
            unnorm_val_imgs = copy.deepcopy(val_imgs)

            # TODO: use de-norm fn
            unnorm_val_imgs = unnorm_val_imgs * torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1) 
            unnorm_val_imgs = unnorm_val_imgs + torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)

            val_labels = to_device(val_dict["label"], device)

            if self.opt.mask_input:
                val_masks = to_device(val_dict["mask"], device)
                H, W = val_imgs.shape[-2:]

                # only masking query input
                val_masks_q = val_masks
                val_masks_t = torch.zeros_like(val_masks)

                square_masks_t = val_masks_t.reshape(-1, H//model.patch_size, W//model.patch_size)
                square_masks_q = val_masks_q.reshape(-1, H//model.patch_size, W//model.patch_size)
                square_masks_q = F.interpolate(square_masks_q.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1).long()
                square_masks_t = F.interpolate(square_masks_t.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1).long()
            else:
                val_masks_t = None
                val_masks_q = None
                square_masks_q = None
                square_masks_t = None


            if self.opt.frozen_target:
                target_seg_masks = model.target_seg_net.get_target_seg_masks(
                                                            val_imgs, 
                                                            include_void=False, 
                                                            high_res=True, 
                                                            masks=val_masks_t, 
                                                            use_sigmoid=self.opt.use_sigmoid, 
                                                            )
            else:
                target_seg_masks = model.seg_net.get_target_seg_masks(
                                                        val_imgs, 
                                                        include_void=True, 
                                                        high_res=True, 
                                                        masks=val_masks_t, 
                                                        use_sigmoid=self.opt.use_sigmoid,
                                                        )

            target_ms_imgs, target_segs = torch.max(target_seg_masks, dim=1)


            query_seg_masks = model.seg_net.get_query_seg_masks(val_imgs, include_void=True, high_res=True, masks=val_masks_q, use_sigmoid=self.opt.use_sigmoid)
            query_seg_masks_unmasked = model.seg_net.get_query_seg_masks(val_imgs, include_void=True, high_res=True, masks=None, use_sigmoid=self.opt.use_sigmoid)

            # NOTE: segs from MASKED query seg masks, ms_imgs from UNMASKED query seg masks
            query_segs = torch.argmax(query_seg_masks, dim=1)
            query_ms_imgs = torch.max(query_seg_masks_unmasked, dim=1)[0]

            ### get the rank of each confidence value in the batch, then norm to [0,1] ###
            # do this to get a scale invariant measure of confidence
            bs, h, w = query_ms_imgs.shape
            query_ms_imgs_ranked = query_ms_imgs.clone()
            query_ms_imgs_ranked = query_ms_imgs_ranked.view(-1).argsort().argsort().view(bs, h, w)
            query_ms_imgs_ranked = query_ms_imgs_ranked.float() / query_ms_imgs_ranked.max()

            target_ms_imgs_ranked = target_ms_imgs.clone()
            target_ms_imgs_ranked = target_ms_imgs_ranked.view(-1).argsort().argsort().view(bs, h, w)
            target_ms_imgs_ranked = target_ms_imgs_ranked.float() / target_ms_imgs_ranked.max()

            for batch_no in range(val_imgs.shape[0]):
                unnorm_img = unnorm_val_imgs[batch_no].permute(1,2,0).detach().cpu().numpy()
                predicted_mask_target = target_segs[batch_no].detach().cpu().numpy()
                predicted_mask_query = query_segs[batch_no].detach().cpu().numpy()

                ms_img_t_quant_ranked = target_ms_imgs_ranked[batch_no].detach().cpu()
                # quantise ms_img_q_quant into N bins, with numbers 0 to N-1
                ms_img_t_quant_ranked = torch.floor(ms_img_t_quant_ranked * 10).long().numpy()

                ms_img_q_quant_ranked = query_ms_imgs_ranked[batch_no].detach().cpu()
                # quantise ms_img_q_quant into N bins, with numbers 0 to N-1
                ms_img_q_quant_ranked = torch.floor(ms_img_q_quant_ranked * 10).long().numpy()


                ground_truth_mask = val_labels[batch_no].detach().cpu().numpy()
                if square_masks_q is not None:
                    square_mask_q = square_masks_q[batch_no].detach().cpu().numpy()
                else:
                    square_mask_q = None
                if square_masks_t is not None:
                    square_mask_t = square_masks_t[batch_no].detach().cpu().numpy()
                else:
                    square_mask_t = None
                # display segmentations as masks in wandb
                masks_log = {}
                masks_log["target_predictions"] = {"mask_data": predicted_mask_target, "class_labels": self.class_dict}
                masks_log["query_predictions"] = {"mask_data": predicted_mask_query, "class_labels": self.class_dict}
                masks_log["ground_truth"] = {"mask_data": ground_truth_mask, "class_labels": self.class_dict}
                masks_log["ms_img_t_ranked"] = {"mask_data": ms_img_t_quant_ranked, "class_labels": {idx : str(idx) for idx in range(11)}}
                masks_log["ms_img_q_ranked"] = {"mask_data": ms_img_q_quant_ranked, "class_labels": {idx : str(idx) for idx in range(11)}}
                if square_mask_q is not None:
                    masks_log["mask_q"] = {"mask_data": square_mask_q, "class_labels": {idx : str(idx) for idx in range(11)}}

                if square_mask_t is not None:
                    masks_log["mask_t"] = {"mask_data": square_mask_t, "class_labels": {idx : str(idx) for idx in range(11)}}
                masked_image = wandb.Image(
                                    unnorm_img,
                                    masks=masks_log,
                                    )
                wandb.log({f"val_segs {dataset_name}/{seg_count}": masked_image}, commit=False)

                seg_count += 1
    ######################################################################################################################################################

    ######################################################################################################################################################
    @torch.no_grad()
    def view_train_segmentations(self, train_dataset, model):
        device = next(model.parameters()).device

        self.train_seg_idxs = [int(idx) for idx in self.train_seg_idxs]
        train_subset = torch.utils.data.Subset(train_dataset, self.train_seg_idxs)

        if self.opt.mask_input:
            from utils.collation_utils import get_collate_fn
            val_collate_fn = get_collate_fn(
                                    img_size=self.model.crop_size, 
                                    patch_size=self.model.patch_size, 
                                    random_mask_prob=self.opt.random_mask_prob, 
                                    min_mask_prob=self.opt.min_mask_prob, 
                                    max_mask_prob=self.opt.max_mask_prob,
                                    )
        else:
            val_collate_fn = None

        _num_workers = 0 if self.opt.num_workers == 0 else 2
        dataloader = torch.utils.data.DataLoader(
                                            dataset=train_subset, 
                                            batch_size=min(self.opt.batch_size, len(train_subset)), 
                                            shuffle=False, 
                                            num_workers=_num_workers, 
                                            drop_last=False,
                                            collate_fn=val_collate_fn,
                                            )
        raw_seg_count = 0
        labelled_seg_count = 0
        log_dict = {}
        iterator = tqdm(dataloader)
        for step, (labelled_dict, raw_dict) in enumerate(iterator):
            ######################################################################################################################################################
            ### LABELLED ###
            labelled_imgs = to_device(labelled_dict["img"], device)
            # un-normalise labelled_imgs
            unnorm_labelled_imgs = copy.deepcopy(labelled_imgs)
            unnorm_labelled_imgs = unnorm_labelled_imgs * torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1) 
            unnorm_labelled_imgs = unnorm_labelled_imgs + torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
            labelled_labels = to_device(labelled_dict["label"], device)

            H, W = labelled_imgs.shape[-2:]

            if self.opt.frozen_target:
                seg_masks = model.target_seg_net.get_target_seg_masks(labelled_imgs, include_void=False, high_res=True)
            else:
                seg_masks = model.seg_net.get_target_seg_masks(labelled_imgs, include_void=True, high_res=True)

            labelled_segs = torch.argmax(seg_masks, dim=1)

            for batch_no in range(labelled_imgs.shape[0]):
                unnorm_labelled_img = unnorm_labelled_imgs[batch_no].permute(1,2,0).detach().cpu().numpy()
                labelled_predicted_mask = labelled_segs[batch_no].detach().cpu().numpy()
                labelled_gt_mask = labelled_labels[batch_no].detach().cpu().numpy()

                masks_log = {}
                masks_log["predictions"] = {"mask_data": labelled_predicted_mask, "class_labels": self.class_dict}
                masks_log["ground_truth"] = {"mask_data": labelled_gt_mask, "class_labels": self.class_dict}
                masked_labelled_image = wandb.Image(
                                    unnorm_labelled_img,
                                    masks=masks_log,
                                    )
                log_dict[f"labelled train segs/{labelled_seg_count}"] =  masked_labelled_image
                wandb.log(log_dict, commit=False)
                labelled_seg_count += 1
            ######################################################################################################################################################

            ######################################################################################################################################################
            ### RAW ###
            if "img_1" in raw_dict.keys():
                raw_imgs_t = to_device(raw_dict["img_1"], device)
                raw_imgs_q = to_device(raw_dict["img_2"], device)
                raw_crop_boxes_A = to_device(raw_dict["box_A"], device)
                raw_crop_boxes_B = to_device(raw_dict["box_B"], device)

                if self.opt.mask_input:
                    val_masks = to_device(raw_dict["mask"], device)
                    H, W = labelled_imgs.shape[-2:]

                    # masking only query input
                    val_masks_t = torch.zeros_like(val_masks)
                    val_masks_q = val_masks

                    # prepare square masks for visualisation
                    square_masks_t = val_masks_t.reshape(-1, H//model.patch_size, W//model.patch_size)
                    square_masks_q = val_masks_q.reshape(-1, H//model.patch_size, W//model.patch_size)
                    square_masks_q = F.interpolate(square_masks_q.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1)
                    square_masks_t = F.interpolate(square_masks_t.float().unsqueeze(1), size=(H,W), mode="nearest").squeeze(1)
                else:
                    val_masks_t = None
                    val_masks_q = None
                    square_masks_q = None
                    square_masks_t = None

                ### target branch ###
                raw_imgs_t_tA = raw_imgs_t
                if self.opt.frozen_target:
                    seg_masks_t_tAB = model.target_seg_net.get_target_seg_masks(
                                                                        raw_imgs_t, 
                                                                        include_void=False, 
                                                                        high_res=True, 
                                                                        masks=val_masks_t, 
                                                                        use_sigmoid=self.opt.use_sigmoid, 
                                                                        )
                else:
                    seg_masks_t_tAB = model.seg_net.get_target_seg_masks(
                                                                raw_imgs_t, 
                                                                include_void=False, 
                                                                high_res=True, 
                                                                masks=val_masks_t, 
                                                                use_sigmoid=self.opt.use_sigmoid, 
                                                                )


                ### query branch ###
                raw_imgs_q_tB = raw_imgs_q
                seg_masks_q_tBA_masked = model.seg_net.get_query_seg_masks(raw_imgs_q, include_void=True, high_res=True, masks=val_masks_q, use_sigmoid=self.opt.use_sigmoid)
                seg_masks_q_tBA_unmasked = model.seg_net.get_query_seg_masks(raw_imgs_q, include_void=True, high_res=True, masks=None, use_sigmoid=self.opt.use_sigmoid)

                # un-normalise raw_imgs
                raw_imgs_t_tA = raw_imgs_t_tA * torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1) 
                raw_imgs_q_tB = raw_imgs_q_tB + torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)

                model.update_gamma(seg_masks_q=seg_masks_q_tBA_unmasked, seg_masks_t=seg_masks_t_tAB)

                if (self.opt.gamma_scaling is None) or (self.opt.gamma_scaling == "None"):
                    gamma_seg_masks_q_tBA = seg_masks_q_tBA_unmasked.clone()
                elif self.opt.gamma_scaling == "softmax":
                    gamma_seg_masks_q_tBA = torch.softmax(seg_masks_q_tBA_unmasked.detach()/self.opt.gamma_temp, dim=1)
                
                gamma_seg_masks_q_tBA = segmasks2gammasegmasks(gamma_seg_masks_q_tBA, gamma=model.gamma.detach(), opt=self.opt)
                gamma_masks_q_tBA = torch.eq(torch.argmax(gamma_seg_masks_q_tBA, dim=1), len(self.class_labels)-1).float()      # 1 where uncertain


                xent_loss_imgs = self.losses.calculate_masking_ssl_loss(
                                                                    seg_masks_t=seg_masks_t_tAB,
                                                                    seg_masks_q=seg_masks_q_tBA_masked,
                                                                    gamma_masks_q=gamma_masks_q_tBA,
                                                                    raw_masks_q=val_masks_q,
                                                                    raw_masks_t=val_masks_t,
                                                                    return_loss_imgs=True,
                                                                    )
                # normalise to [0,1]
                xent_loss_imgs = (xent_loss_imgs - xent_loss_imgs.min()) / (xent_loss_imgs.max() - xent_loss_imgs.min())


                ms_imgs_t, segs_t_tAB = torch.max(seg_masks_t_tAB, dim=1)
                # NOTE segs from masked
                segs_q_tBA = torch.argmax(seg_masks_q_tBA_masked, dim=1)
                # NOTE ms imgs from unmasked
                ms_imgs_q = torch.max(seg_masks_q_tBA_unmasked, dim=1)[0]

                ### get the rank of each confidence value in the batch, then norm to [0,1]###
                # do this to get a scale invariant measure of confidence
                bs, h, w = ms_imgs_q.shape
                ms_imgs_q_ranked = ms_imgs_q.view(-1).argsort().argsort().view(bs, h, w)
                ms_imgs_q_ranked = ms_imgs_q_ranked.float() / ms_imgs_q_ranked.max()

                ms_imgs_t_ranked = ms_imgs_t.view(-1).argsort().argsort().view(bs, h, w)
                ms_imgs_t_ranked = ms_imgs_t_ranked.float() / ms_imgs_t_ranked.max()

                consistency_masks = torch.eq(segs_t_tAB, segs_q_tBA).float()

                for batch_no in range(raw_imgs_t_tA.shape[0]):
                    raw_img_t_tA = raw_imgs_t_tA[batch_no].permute(1,2,0).detach().cpu().numpy()
                    raw_predicted_mask_t = segs_t_tAB[batch_no].detach().cpu().numpy()

                    raw_img_q_tB = raw_imgs_q_tB[batch_no].permute(1,2,0).detach().cpu().numpy()
                    raw_predicted_mask_q = segs_q_tBA[batch_no].detach().cpu().numpy()
                    consistency_mask = consistency_masks[batch_no].detach().cpu().numpy()
                    # ms_imgs

                    ms_img_q_quant_ranked = ms_imgs_q_ranked[batch_no].detach().cpu()
                    # quantise ms_img_q_quant into N bins, with numbers 0 to N-1
                    ms_img_q_quant_ranked = torch.floor(ms_img_q_quant_ranked * 10).long().numpy()

                    ms_img_t_quant_ranked = ms_imgs_t_ranked[batch_no].detach().cpu()
                    # quantise ms_img_q_quant into N bins, with numbers 0 to N-1
                    ms_img_t_quant_ranked = torch.floor(ms_img_t_quant_ranked * 10).long().numpy()


                    gamma_mask = gamma_masks_q_tBA[batch_no].detach().cpu().numpy()

                    xent_loss_img = xent_loss_imgs[batch_no].detach().cpu()
                    # quantise xent_loss_img into N bins, with numbers 0 to N-1
                    xent_loss_img = torch.ceil(xent_loss_img * 10).long().numpy()

                    if square_masks_q is not None:
                        square_mask_q = square_masks_q[batch_no].detach().cpu().numpy()
                    else:
                        square_mask_q = None
        
                    if square_masks_t is not None:
                        square_mask_t = square_masks_t[batch_no].detach().cpu().numpy()
                    else:
                        square_mask_t = None

                    ### target ###
                    masks_log_t = {}
                    # 1) segs
                    masks_log_t["predictions_t"] = {"mask_data": raw_predicted_mask_t, "class_labels": self.class_dict}
                    # 2) ms_imgs
                    masks_log_t["ms_img_t_ranked"] = {"mask_data": ms_img_t_quant_ranked, "class_labels": {idx : str(idx) for idx in range(11)}}
                    # 3) masks
                    if square_mask_t is not None:
                        masks_log_t["mask_t"] = {"mask_data": square_mask_t, "class_labels": {idx : str(idx) for idx in range(11)}}
                    # 5) consistency_masks
                    masks_log_t["consistency_mask"] = {"mask_data": consistency_mask}
                    # 6) xent_loss
                    masks_log_t["xent_loss"] = {"mask_data": xent_loss_img}
                    # 7) gamma_masks
                    masks_log_t["gamma_mask"] = {"mask_data": gamma_mask}

                    ### query ###
                    masks_log_q = {}
                    # 1) segs
                    masks_log_q["predictions_q"] = {"mask_data": raw_predicted_mask_q, "class_labels": self.class_dict}
                    # 2) ms_imgs
                    masks_log_q["ms_img_q_ranked"] = {"mask_data": ms_img_q_quant_ranked, "class_labels": {idx : str(idx) for idx in range(11)}}
                    # 3) masks
                    if square_mask_q is not None:
                        masks_log_q["mask_q"] = {"mask_data": square_mask_q, "class_labels": {idx : str(idx) for idx in range(11)}}
                    # 5) consistency_masks
                    masks_log_q["consistency_mask"] = {"mask_data": consistency_mask}
                    # 6) xent_loss
                    masks_log_q["xent_loss"] = {"mask_data": xent_loss_img}
                    # 7) gamma_masks
                    masks_log_q["gamma_mask"] = {"mask_data": gamma_mask}
                    
                    masked_raw_image_t = wandb.Image(
                                        raw_img_t_tA,
                                        masks=masks_log_t
                                        )
                    masked_raw_image_q = wandb.Image(
                                        raw_img_q_tB,
                                        masks=masks_log_q
                                        )

                    log_dict[f"raw train segs/{raw_seg_count}_target"] = masked_raw_image_t
                    log_dict[f"raw train segs/{raw_seg_count}_query"] = masked_raw_image_q
                    wandb.log(log_dict, commit=False)
                    raw_seg_count += 1
            ######################################################################################################################################################


    ######################################################################################################################################################
    @torch.no_grad()
    def validate_uncertainty_estimation(self, val_dataset, model, full_validation_count):
        print("\nValidating uncertainty estimation on", val_dataset.name)
        device = next(model.parameters()).device

        ### init ###
        val_metrics_totals_query, val_metrics_counts_query = init_val_ue_metrics(self.opt.num_thresholds, counts=True)
        val_metrics_totals_target, val_metrics_counts_target = init_val_ue_metrics(self.opt.num_thresholds, counts=True)

        if self.opt.mask_input:
            from utils.collation_utils import get_val_collate_fn
            val_collate_fn = get_val_collate_fn(
                                    img_size=val_dataset.crop_sizes, 
                                    patch_size=model.patch_size, 
                                    random_mask_prob=self.opt.random_mask_prob, 
                                    t_and_q_masks=self.opt.mask_both,
                                    min_mask_prob=self.opt.min_mask_prob, 
                                    max_mask_prob=self.opt.max_mask_prob,
                                    )
        else:
            val_collate_fn = None

        _batch_size = self.opt.val_batch_size if self.opt.val_batch_size is not None else self.opt.batch_size
        full_val_dataloader = torch.utils.data.DataLoader(
                                                    dataset=val_dataset, 
                                                    batch_size=_batch_size, 
                                                    shuffle=False, 
                                                    num_workers=2, 
                                                    drop_last=False,
                                                    collate_fn=val_collate_fn,
                                                    )

        ### calculate certainty metrics ###
        iterator = tqdm(full_val_dataloader)

        for step, (val_dict) in enumerate(iterator):
            val_imgs = to_device(val_dict["img"], device)
            val_labels = to_device(val_dict["label"], device)

            # calculate ue metrics
            val_metrics_query = perform_batch_ue_validation(
                                                val_imgs, 
                                                val_labels,
                                                model,
                                                self.opt,
                                                branch="query",
                                                )
            
            val_metrics_target = perform_batch_ue_validation(
                                                val_imgs, 
                                                val_labels,
                                                model,
                                                self.opt,
                                                branch="target",
                                                )

            for key in val_metrics_totals_query:
                val_metrics_totals_query[key] += val_metrics_query[key].sum(0).cpu()
                if len(val_metrics_query[key].shape) > 0:
                    val_metrics_counts_query[key] += val_metrics_query[key].shape[0]
                else:
                    val_metrics_counts_query[key] += 1

            for key in val_metrics_totals_target:
                val_metrics_totals_target[key] += val_metrics_target[key].sum(0).cpu()
                if len(val_metrics_target[key].shape) > 0:
                    val_metrics_counts_target[key] += val_metrics_target[key].shape[0]
                else:
                    val_metrics_counts_target[key] += 1
            ### end of validation loop ###

        ### calculate metrics over dataset ###
        processed_metrics_query = calculate_metrics_suite(val_metrics_totals_query, val_metrics_counts_query)
        processed_metrics_target = calculate_metrics_suite(val_metrics_totals_target, val_metrics_counts_target)
        
        ### plotting metrics to wandb ###
        plot_val_ue_metrics(processed_metrics_query, full_validation_count, dataset_name=val_dataset.name+" (Query)", plot_plots=False)
        plot_val_ue_metrics(processed_metrics_target, full_validation_count, dataset_name=val_dataset.name+" (Target)", plot_plots=False)
    ######################################################################################################################################################