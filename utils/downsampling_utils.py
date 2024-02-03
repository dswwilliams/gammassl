import torch
import torch.nn.functional as F

THING_CLASSES_WEIGHT = 10

class ClassWeightedModalDownSampler(torch.nn.Module):
    """
    Implements class weighted modal downsampling.
        i.e. for a kernel size of N, the value returned for each N x N patch is the mode of the classes in that patch.
        This modal value is also weighted by the class weights, where 'thing' classes are weighted more heavily.
        This means that the mode of the classes in a patch is more likely to be a 'thing' class, than a 'stuff' class.
    This is used to downsample segmentation labels while accounting for class imbalance (many more stuff pixels than thing pixels).
    """
    def __init__(self, known_class_list):
        super().__init__()

        known_class_list = known_class_list
        self.num_classes = len(known_class_list) + 1

        # defining thing classes (in contrast to stuff classes)
        thing_classes = ["pole", "traffic_light", "traffic_sign", "person", 
                                "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        # converting thing class strings to indices
        thing_classes = [known_class_list.index(c) for c in thing_classes]

        # defining class weigjts
        class_weights = torch.ones(self.num_classes)
        class_weights[thing_classes] = THING_CLASSES_WEIGHT
        self.class_weights = class_weights.view(1,1,1,-1)

    def forward(self, labels, dsf=8):
        """
        Args:
            labels (torch.Tensor): size: [bs, H, W], where labels[i,j,k] is the class of the pixel at position (j,k) in image i.
            dsf (int): The factor by which to downsample the labels (Down-Sample Factor)
        Returns:
            modes (torch.Tensor): size: [bs, H//downsample_factor, W//downsample_factor], 
            where modes[i,j,k] is the mode of the classes in the patch of size downsample_factor x downsample_factor at position (j,k) in image i.
        """
        device = labels.device

        _, H, W = labels.size()     # size: [bs, H, W]
        labels = labels.unsqueeze(1).float()    # size: [bs, 1, H, W]

        # transform labels to patches of size (dsf x dsf) (of which there are [bs * (H*W)//(dsf**2)])
        small_labels = F.unfold(labels, kernel_size=dsf, stride=dsf)    # size: [bs, dsf**2, (H*W)//(dsf**2)]
        # convert to one-hot
        small_one_hot_labels = F.one_hot(small_labels.long(), num_classes=self.num_classes)     # size: [bs, dsf**2, (H*W)//(dsf**2), num_classes]
        # count the number of each class in each patch
        class_counts = torch.sum(small_one_hot_labels, dim=1, keepdim=True)      # size: [bs, 1, (H*W)//(dsf**2), num_classes]
        # weight the class counts
        weighted_class_counts = class_counts * self.class_weights.to(device)
        # get the mode of the classes in each patch
        modes = torch.argmax(weighted_class_counts, dim=3)          # size: [bs, 1, (H*W)//(dsf**2)]
        # reshape to bs * H//dsf * W//dsf
        modes = F.fold(modes.float(), kernel_size=1, stride=1, output_size=(H//dsf, W//dsf))     # size: [bs, 1, H//dsf, W//dsf]
        modes = modes.squeeze(1).long()
        return modes



if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)

    known_class_list=["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    class_weighted_modal_downsampler = ClassWeightedModalDownSampler(known_class_list)

    labels = torch.randint(0, 20, (1, 16, 16))
    print("labels.shape:\n", labels)
    downsampled_labels = class_weighted_modal_downsampler(labels, dsf=4)
    print("downsampled_labels.shape:\n", downsampled_labels)