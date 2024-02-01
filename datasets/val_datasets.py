import os
import cv2
import torch
from utils.dataset_utils import central_crop_img, get_img_size_from_aspect_ratio, ImgColourTransform

cityscapes_train_dirs = ["train/jena/", "train/zurich/", "train/weimar/", "train/ulm/", "train/tubingen/", "train/stuttgart/",
              "train/strasbourg/", "train/monchengladbach/", "train/krefeld/", "train/hanover/",
              "train/hamburg/", "train/erfurt/", "train/dusseldorf/", "train/darmstadt/", "train/cologne/",
              "train/bremen/", "train/bochum/", "train/aachen/"]
cityscapes_val_dirs = ["val/frankfurt/", "val/munster/", "val/lindau/"]
cityscapes_test_dirs = ["test/berlin/", "test/bielefeld/", "test/bonn/", "test/leverkusen/", "test/mainz/", "test/munich/"]


def normalize_img(img, imagenet=False):
    if imagenet:
        # normalize the img (with the mean and std for the pretrained ResNet):
        if (img > 2).any():
            img = img/255.0
        img = img - torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        img = img/torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2) # (shape: (256, 256, 3))
        img = img.float()
    else:
        # normalize the img (with the mean and std for the pretrained ResNet):
        if (img > 2).any():
            img = img/255.0
        # [0,1] -> [-1, 1]
        img = (img - 0.5) * 2
        img = img.float()
    return img


def get_preprocessed_data(example, imagenet_norm, resize_sizes, crop_sizes, colour_transform=None, patch_size=None):
    img = cv2.cvtColor(cv2.imread(example["img_path"]), cv2.COLOR_BGR2RGB)
    label_img = cv2.imread(example["label_path"], cv2.IMREAD_GRAYSCALE)

    # resizing and crop
    img = cv2.resize(img, (resize_sizes[1], resize_sizes[0]), interpolation=cv2.INTER_NEAREST)
    label_img = cv2.resize(label_img, (resize_sizes[1], resize_sizes[0]), interpolation=cv2.INTER_NEAREST)
    # central crop to make width divisible by 2*patch_size
    if patch_size is not None:
        img, label_img = central_crop_img(img, label=label_img, output_shape=(crop_sizes[0], crop_sizes[1]))

    ### converting numpy -> torch ###
    img = torch.from_numpy(img) 
    img = img.permute(2,0,1).float()/255
    if colour_transform is not None:
        img = colour_transform(img)
    img = normalize_img(img, imagenet=imagenet_norm)
    label_img = torch.from_numpy(label_img).long()

    return img, label_img

def get_cityscapes_examples(dataroot):
    cityscapes_img_dir = os.path.join(dataroot, "cityscapes/leftImg8bit/")
    cityscapes_label_dir = os.path.join(dataroot, "cityscapes/meta/label_imgs/")
    examples = []
    for val_dir in cityscapes_val_dirs:
        val_img_dir_path = cityscapes_img_dir + val_dir

        file_names = os.listdir(val_img_dir_path)
        for file_name in file_names:
            img_id = file_name.split("_leftImg8bit.png")[0]
            img_path = val_img_dir_path + file_name
            label_img_path = cityscapes_label_dir + img_id + ".png"
            example = {}
            example["name"] = "cityscapes"
            example["img_path"] = img_path
            example["label_path"] = label_img_path
            examples.append(example)
    return examples

def get_bdd_examples(dataroot):
    examples = []
    img_dataroot = os.path.join(dataroot, "images", "10k")
    label_dataroot = os.path.join(dataroot, "labels", "sem_seg", "masks")
    for dataset_type in os.listdir(img_dataroot):
        if dataset_type in ["val"]:
            for img_name in os.listdir(os.path.join(img_dataroot, dataset_type)):
                if img_name[-4:] == ".jpg":
                    example = {}
                    example["name"] = img_name[:-4]
                    example["img_path"] = os.path.join(img_dataroot, dataset_type, img_name)
                    example["label_path"] = os.path.join(label_dataroot, dataset_type, img_name.replace(".jpg", ".png"))
                    examples.append(example)
    return examples


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, name, dataroot, use_imagenet_norm, val_transforms=False, patch_size=None):
        
        self.name = name

        # Cityscapes
        if self.name == "CityscapesVal":
            self.aspect_ratio = (1024, 2048)      # (H,W)
            self.examples = get_cityscapes_examples(dataroot)
        # BDD
        elif self.name == "BDDVal":
            self.aspect_ratio = (720, 1280)      # (H,W)
            self.examples = get_bdd_examples(dataroot)

        self.imagenet_norm = use_imagenet_norm
        self.patch_size = patch_size
        # resize_sizes are intermediate, crop_sizes are the final size
        self.resize_sizes, self.crop_sizes = get_img_size_from_aspect_ratio(self.aspect_ratio, patch_size=self.patch_size)

        if val_transforms:
            self.colour_transform = ImgColourTransform(n_seq_transforms=1)
        else:
            self.colour_transform = None

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img, label_img = get_preprocessed_data(
                                        example=example, 
                                        imagenet_norm=self.imagenet_norm, 
                                        resize_sizes=self.resize_sizes,
                                        crop_sizes=self.crop_sizes, 
                                        colour_transform=self.colour_transform, 
                                        patch_size=self.patch_size,
                                        )

        output = {}
        output["img"] = img
        output["label"] = label_img
        return output

    def __len__(self):
        return self.num_examples