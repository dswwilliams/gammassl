
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")

class Extract_HyperSpherePrototypes(nn.Module):
    def __init__(self, num_known_classes):
        super(Extract_HyperSpherePrototypes, self).__init__()
        self.num_known_classes = num_known_classes  
    
    def forward(self, features, labels, output_metrics=False):
        """
        inputs:
        features.shape = [bs, feature_length, h, w]
        labels.shape = [bs, h, w]
        
        outputs:
        prototypes.shape = [feature_length, num_classes]
        """

        bs, feature_length, h, w = features.shape

        ### prepare features ###
        features = F.normalize(features, p=2, dim=1)
        features = features.permute(0,2,3,1).reshape(bs*h*w, feature_length)        # shape: [bs*h*w, feature_length]

        ### prepare labels ###
        labels = labels.reshape(bs*h*w)                # shape: [bs*h*w]
        one_hot_labels = F.one_hot(labels, num_classes=self.num_known_classes+1).float()          # shape: [bs*h*w, num_known_classes+1]

        ### calculate prototypes ### 
        prototypes = torch.matmul(features.T, one_hot_labels)         # shape: [feature_length, num_known_classes+1]
        prototypes = prototypes[:,:-1]                            # shape: [feature_length, num_known_classes]
        prototypes = F.normalize(prototypes, dim=0, p=2)          # shape: [feature_length, num_known_classes]

        if output_metrics:
            with torch.no_grad():
                # calculate mean sim for each prototype cluster
                sim_w_prototypes = torch.matmul(features.detach(), prototypes.detach())                         # shape: [bs*h*w, num_known_classes]
                one_hot_labels = one_hot_labels[:,:-1]                                         # shape: [bs*h*w, num_known_classes]
                class_count = one_hot_labels.sum(0)
                class_count[class_count == 0] = 1
                mean_sim_w_GTprototypes = (one_hot_labels * sim_w_prototypes).sum(0) / class_count  # shape: [num_known_classes]
                return prototypes, mean_sim_w_GTprototypes
        else:
            return prototypes

class Segment_via_HyperSpherePrototypes(nn.Module):
    def __init__(self):
        super(Segment_via_HyperSpherePrototypes, self).__init__()

    def forward(self, features, global_prototypes, gamma=None, visualiser=None, output_metrics=False):
        bs, feature_length, h, w = features.shape           # shape: [bs, feature_length, h, w]

        # normalise features to compute cosine similarity
        features = F.normalize(features, p=2, dim=1)

        global_prototypes = F.normalize(global_prototypes, p=2, dim=0)
        num_known_classes = global_prototypes.shape[1]

        features = features.permute(0,2,3,1).reshape(-1, feature_length)        # shape: [bs*h*w, feature_length]

        seg_masks = torch.matmul(features, global_prototypes)        # shape: [bs*h*w, num_known_classes]

        if output_metrics:
            with torch.no_grad():
                ### calculating mean and std of similarity between features their NN prototype ###
                sim_to_NNprototype, segs = torch.max(seg_masks.detach(), dim=1)        # shapes: [bs*h*w], [bs*h*w]
                segs = F.one_hot(segs, num_classes=num_known_classes)                    # shape: [bs*h*w, num_known_classes]
                # checking how many pixels are segmented as each of the classes
                class_count = segs.sum(0)           # shape: [num_known_classes]
                # if no pixels are segmented as a given class, then give class count as 1 to avoid NaN (the mean sim = 0 -> 0/1 = 0 instead of NaN)
                class_count[class_count == 0] = 1
                class_masked_sim_to_NNprototype = (segs * seg_masks)

                mean_sim_per_class_to_NNprototype = class_masked_sim_to_NNprototype.sum(0) / class_count
                # variance_per_class_to_NNprototype = (segs * (class_masked_sim_to_NNprototype - mean_sim_per_class_to_NNprototype)**2).sum(0) / (class_count)

        if visualiser is not None:
            with torch.no_grad():
                vis = visualiser["vis"]
                vis.histogram(sim_to_NNprototype, 
                win=visualiser["name"], 
                env="mean-sim-to-NN-prototypes",
                opts=dict(
                        xlabel='CosineSim',
                        numbins=100,
                        title=visualiser["name"]
                    ))

        if gamma is not None:
            if gamma.requires_grad:
                gammas = gamma * torch.ones(bs*h*w, 1, requires_grad=True).to(features.device)      # shape: [bs*h*w, 1]
            else:
                gammas = gamma * torch.ones(bs*h*w, 1, requires_grad=False).to(features.device)      # shape: [bs*h*w, 1]
            seg_masks = torch.cat((seg_masks, gammas), dim=1)             # shape: [bs*h*w, num_known_classes+1]


        seg_masks = seg_masks.reshape(bs, h, w, -1).permute(0,3,1,2)     # shape: [bs, num_known_classes, h, w]

        if output_metrics:
            return seg_masks, mean_sim_per_class_to_NNprototype
        else:
            return seg_masks, None


def calc_inter_prototype_sims(prototypes):
    """
    prototypes.shape = [feature_length, num_known_classes]
    """
    
    sim_matrix = torch.matmul(prototypes.T, prototypes)         # shape: [num_known_classes, num_known_classes]

    mask = (1 - torch.eye(sim_matrix.shape[0])).to(sim_matrix.device)
    mean_interclass_sim = (mask * sim_matrix).sum(0) / mask.sum(0)

    max_interclass_sim = (mask * sim_matrix).max(0)[0]
    min_interclass_sim = (mask * sim_matrix).min(0)[0]

    return max_interclass_sim, mean_interclass_sim, min_interclass_sim


def calculate_cosine_sim_threhold(seg_masks, proportion_to_reject):
    with torch.no_grad():
        # get vector of cosine sims for each pixel
        sims_per_pixel = torch.max(seg_masks.detach(), dim=1)[0]         # shape [bs, h, w]
        sims_per_pixel = sims_per_pixel.flatten()
        sims_per_pixel = torch.sort(sims_per_pixel, descending=False)[0]
        num_pixels_to_reject = int(proportion_to_reject * sims_per_pixel.numel())
        # find the value of cosine sim that would reject the stated proportion
        cosine_sim_threshold = sims_per_pixel[num_pixels_to_reject]
    return cosine_sim_threshold


"""
- Using the SAX data, we can compute feature space statistics to evaluate how the training is progressing.
- This could be very useful in the development of the techniques
"""

# def calculate_feature_stats(seg_masks, labels):
#     """
#     - calculate average similarity with prototype and the variability in this similarity
#     """
#     return mean_sim_to_GTprototype, var_sim_to_GTprototype

def calc_mean_sim_w_GTprototypes(features, prototypes, labels, num_known_classes):
    with torch.no_grad():
        bs, feature_length, h, w = features.shape
        features = F.normalize(features, p=2, dim=1)
        features = features.permute(0,2,3,1).reshape(bs*h*w, feature_length)        # shape: [bs*h*w, feature_length]
        labels = labels.reshape(bs*h*w)                # shape: [bs*h*w]
        one_hot_labels = F.one_hot(labels, num_classes=num_known_classes+1).float()          # shape: [bs*h*w, num_known_classes+1]
        # calculate mean sim for each prototype cluster
        sim_w_prototypes = torch.matmul(features.detach(), prototypes.detach())                           # shape: [bs*h*w, num_known_classes]
        one_hot_labels = one_hot_labels[:,:-1]                                                      # shape: [bs*h*w, num_known_classes]
        class_count = one_hot_labels.sum(0)
        class_count[class_count == 0] = 1
        mean_sim_w_GTprototypes = (one_hot_labels * sim_w_prototypes).sum(0) / class_count  # shape: [num_known_classes]
        return mean_sim_w_GTprototypes