import torch
import torch.nn.functional as F
import pickle
import io


class CPU_Unpickler(pickle.Unpickler):
    """
    - deals with issue of pickling device on cuda
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def read_from_pickle(path):
    try:
        while True:
            yield CPU_Unpickler(open(path, 'rb')).load()
    except EOFError:
        pass


proto_save_path = "/Volumes/scratchdata/dw/gamma_star_gbranch_t/prototypes_6.pkl"


def load_prototypes_from_pkl(proto_save_path, device):
    prototypes = next(read_from_pickle(proto_save_path))
    if device is not None:
        prototypes = prototypes.to(device)
    return prototypes
        

def extract_prototypes(features, labels, num_known_classes=19, output_metrics=False):
    """
    Extract prototypes from input features based on labels.

    Args:
        features (torch.Tensor): Input feature maps with shape [bs, feature_length, h, w].
        labels (torch.Tensor): Ground-truth labels with shape [bs, h, w], i.e same spatial size as features.
        num_known_classes (int, optional): Number of known classes (excluding void class). Default is 19.
        output_metrics (bool, optional): Whether to output additional metrics. Default is False.

    Returns:
        torch.Tensor: Prototypes for known classes with shape [feature_length, num_known_classes].

    If 'output_metrics' is True, the function also returns mean similarity values
        between features and prototypes for each known class.
    """
    bs, feature_length, h, w = features.shape

    # prepare features
    features = F.normalize(features, p=2, dim=1)
    features = features.permute(0,2,3,1).reshape(bs*h*w, feature_length)        # shape: [bs*h*w, feature_length]

    # prepare labels
    labels = labels.reshape(bs*h*w)                # shape: [bs*h*w]
    one_hot_labels = F.one_hot(labels, num_classes=num_known_classes+1).float()          # shape: [bs*h*w, num_known_classes+1]

    # calculate prototypes, ignoring void class
    prototypes = torch.matmul(features.T, one_hot_labels)         # shape: [feature_length, num_known_classes+1]
    prototypes = prototypes[:,:-1]                            # shape: [feature_length, num_known_classes]
    prototypes = F.normalize(prototypes, dim=0, p=2)          # shape: [feature_length, num_known_classes]

    if output_metrics:
        return prototypes, calculate_mean_prototype_sim(features, prototypes, one_hot_labels)
    else:
        return prototypes


@torch.no_grad()
def calculate_mean_prototype_sim(features, prototypes, one_hot_labels):
    # flatten inputs
    if one_hot_labels.shape > 2:
        one_hot_labels = one_hot_labels.reshape(-1, one_hot_labels.shape[-1])          # shape: [bs*h*w, num_known_classes]
    if features.shape > 2:
        features = features.reshape(-1, features.shape[-1])          # shape: [bs*h*w, feature_length]

    # ignoring void class
    num_known_classes = prototypes.shape[-1]
    if one_hot_labels.shape[-1] > num_known_classes:
        one_hot_labels = one_hot_labels[:,:num_known_classes]            # shape: [bs*h*w, num_known_classes]

    # calculate mean sim for each prototype cluster
    sim_w_prototypes = torch.matmul(features.detach(), prototypes.detach())                         # shape: [bs*h*w, num_known_classes]
    class_count = one_hot_labels.sum(0)
    class_count[class_count == 0] = 1
    mean_sim_w_GTprototypes = (one_hot_labels * sim_w_prototypes).sum(0) / class_count  # shape: [num_known_classes]
    return mean_sim_w_GTprototypes


def segment_via_prototypes(features, prototypes, gamma=None, output_metrics=False):
    """
    Assign features to semantic classes based on cosine similarity to prototypes.

    Args:
        features (torch.Tensor): Input feature maps with shape [bs, feature_length, h, w].
        prototypes (torch.Tensor): Prototypes for known classes with shape [feature_length, num_known_classes].
        gamma (float, optional): Confidence threshold. Default is None.
        output_metrics (bool, optional): Whether to output additional metrics. Default is False.

    Returns:
        torch.Tensor: Segmentation masks with shape [bs, num_known_classes, h, w].
    """
    bs, feature_length, h, w = features.shape           # shape: [bs, feature_length, h, w]

    # normalise features to compute cosine similarity
    features = F.normalize(features, p=2, dim=1)

    prototypes = F.normalize(prototypes, p=2, dim=0)
    num_known_classes = prototypes.shape[1]

    features = features.permute(0,2,3,1).reshape(-1, feature_length)        # shape: [bs*h*w, feature_length]

    seg_masks = torch.matmul(features, prototypes)        # shape: [bs*h*w, num_known_classes]

    if output_metrics:
        # calculate the mean similarity between features and their nearest prototype
        mean_sim_to_NNprototype = calculate_mean_prototype_sim(features, prototypes, one_hot_labels=F.one_hot(seg_masks.argmax(1), num_known_classes))

    if gamma is not None:
        if gamma.requires_grad:
            gammas = gamma * torch.ones(bs*h*w, 1, requires_grad=True).to(features.device)      # shape: [bs*h*w, 1]
        else:
            gammas = gamma * torch.ones(bs*h*w, 1, requires_grad=False).to(features.device)      # shape: [bs*h*w, 1]
        seg_masks = torch.cat((seg_masks, gammas), dim=1)             # shape: [bs*h*w, num_known_classes+1]

    seg_masks = seg_masks.reshape(bs, h, w, -1).permute(0,3,1,2)     # shape: [bs, num_known_classes, h, w]

    if output_metrics:
        return seg_masks, mean_sim_to_NNprototype
    else:
        return seg_masks


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