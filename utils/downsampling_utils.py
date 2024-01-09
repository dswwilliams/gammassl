import torch
import torch.nn.functional as F

class ClassWeightedModalDownSampler(torch.nn.Module):
    def __init__(self, known_class_list):
        super().__init__()
        self.known_class_list = known_class_list

        self.num_classes = len(known_class_list) + 1
        if self.num_classes == 20:
            thing_classes = ["pole", "traffic_light", "traffic_sign","person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        elif self.num_classes == 38:
            thing_classes = ["cabinet", "bed", "chair", "sofa", "table", "door", "bookshelf", "picture", "counter", 
                                "blinds", "desk", "shelves", "dresser", "pillow", "mirror", "floor_mat", "clothes",
                                "books", "fridge", "tv", "paper", "towel", "shower_curtain", "box",  "person", "night_stand", 
                                "toilet", "sink", "lamp", "bathtub", "bag"]
        elif self.num_classes == 14:
            thing_classes = ["Bed", "Books", "Chair", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Window"]
        self.thing_classes = [self.known_class_list.index(c) for c in thing_classes]

    def forward(self, labels, downsample_factor=8):

        _, H, W = labels.size()     # size: [bs, H, W]
        labels = labels.unsqueeze(1).float()
        class_weights = torch.ones(self.num_classes).to(labels.device)
        class_weights[self.thing_classes] = 10
        class_weights = class_weights.view(1,1,1,-1)

        small_labels = F.unfold(labels, kernel_size=downsample_factor, stride=downsample_factor)    # size: [bs, downsample_factor**2, (H*W)//(downsample_factor**2)]
        small_one_hot_labels = F.one_hot(small_labels.long(), num_classes=self.num_classes)     # size: [bs, downsample_factor**2, (H*W)//(downsample_factor**2), num_classes]
        class_counts = torch.sum(small_one_hot_labels, dim=1, keepdim=True)      # size: [bs, 1, (H*W)//(downsample_factor**2), num_classes]
        weighted_class_counts = class_counts * class_weights
        modes = torch.argmax(weighted_class_counts, dim=3)          # size: [bs, 1, (H*W)//(downsample_factor**2)]
        modes = F.fold(modes.float(), kernel_size=1, stride=1, output_size=(H//downsample_factor, W//downsample_factor))      # size: [bs, 1, H//downsample_factor, W//downsample_factor]
        return modes.squeeze().long()



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    import sys
    sys.path.append("../")
    from utils.downsampling_utils import class_weighted_modal_downsampling

    labels = torch.tensor(np.random.randint(0, 20, size=(2, 224, 224)))
    print("labels.shape: ", labels.shape)
    downsampled_labels = class_weighted_modal_downsampling(labels, downsample_factor=14, thing_classes=[5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18], num_classes=20)
    print("downsampled_labels.shape: ", downsampled_labels.shape)
    # plt.figure()
    # plt.imshow(labels[0])
    # plt.figure()
    # plt.imshow(downsampled_labels[0])
    # plt.show()