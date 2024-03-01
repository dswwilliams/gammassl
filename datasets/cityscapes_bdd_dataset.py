import torch
import numpy as np
import os
import sys
import cv2
sys.path.append("../")
from utils.dataset_utils import get_initial_scaling_values, random_flip, normalize_img_tensor, ImgColourTransform
from utils.dataset_utils import get_random_crop, resize_data, get_resize_noise
from utils.candr_utils import get_random_crop_boxes

cityscapes_train_dirs = ["train/jena/", "train/zurich/", "train/weimar/", "train/ulm/", "train/tubingen/", "train/stuttgart/",
              "train/strasbourg/", "train/monchengladbach/", "train/krefeld/", "train/hanover/",
              "train/hamburg/", "train/erfurt/", "train/dusseldorf/", "train/darmstadt/", "train/cologne/",
              "train/bremen/", "train/bochum/", "train/aachen/"]

# definitions
DOWNSAMPLE_FACTOR = 2
RESIZE_NOISE_FACTOR = 1.5

class CityscapesxBDDDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset which generates pairs of labelled and unlabelled data from the Cityscapes and BDD datasets.
    """
    def __init__(self, labelled_dataroot, 
                        bdd_dataroot, 
                        no_appearance_transform=False, 
                        min_crop_ratio=1.2, 
                        max_crop_ratio=3, 
                        add_resize_noise=True,
                        only_labelled=False,
                        use_imagenet_norm=True,
                        no_colour=False,
                        crop_size=256,
                        ):

        self.crop_size = crop_size
        self.identity_crop_box = torch.tensor([[0,0], [self.crop_size-1, 0], [self.crop_size-1, self.crop_size-1], [0, self.crop_size-1]])
        self.downsample_factor = DOWNSAMPLE_FACTOR
        self.resize_noise_factor = RESIZE_NOISE_FACTOR
        self.use_imagenet_norm = use_imagenet_norm
        self.no_colour = no_colour
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.add_resize_noise = add_resize_noise
        self.only_labelled = only_labelled
        self.no_appearance_transform = no_appearance_transform

        self.colour_transform = ImgColourTransform(n_seq_transforms=1, no_colour=self.no_colour)

        # setting up unlabelled BDD data
        self.bdd_examples = []
        dataset_root = os.path.join(bdd_dataroot,"images", "100k")
        for dataset_type in os.listdir(dataset_root):
            if dataset_type in ["train"]:
                for img_name in os.listdir(os.path.join(dataset_root, dataset_type)):
                    if img_name[-4:] == ".jpg":
                        img_path = os.path.join(dataset_root, dataset_type, img_name)
                        self.bdd_examples.append(img_path)
        self.num_bdd_examples = len(self.bdd_examples)

        # setting up labelled Cityscapes data
        self.cityscapes_img_dir = os.path.join(labelled_dataroot, "cityscapes/leftImg8bit/")
        self.cityscapes_label_dir = os.path.join(labelled_dataroot, "cityscapes/meta/label_imgs/")
        self.examples = []
        for train_dir in cityscapes_train_dirs:
            train_img_dir_path = self.cityscapes_img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.cityscapes_label_dir + img_id + ".png"
                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                self.examples.append(example)

        self.num_labelled_examples = len(self.examples)

    def preprocess_data(self, img, label=None):
        """ Preprocesses the data by resizing, cropping, and flipping the image and label """

        # resizing
        h, w,_ = img.shape
        h, w = get_initial_scaling_values(h, w, downsample_factor=self.downsample_factor)
        if self.add_resize_noise:
            # add noise to resize (as a proportion of how much bigger height is than crop_size)
            noise = get_resize_noise(h, self.crop_size, noise_factor=self.resize_noise_factor)
            w = int((w/h) * noise * self.crop_size)
            h = int(noise * self.crop_size)
        img, label = resize_data(img=img, label=label, new_height=h, new_width=w)

        # cropping
        img, label = get_random_crop(img, label, self.crop_size)

        # flipping 
        img, label = random_flip(img, label, p=0.5)

        # convert numpy -> torch:
        img = torch.from_numpy(img).permute(2,0,1) / 255
        
        if label is not None:
            return img, torch.from_numpy(label)
        else:
            return img

    def __getitem__(self, index):

        # reading in labelled Cityscapes data
        example = np.random.choice(self.examples)
        img_path = example["img_path"]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)

        img, label_img = self.preprocess_data(img, label_img)
        
        # transforming appearance
        if self.no_appearance_transform:
            pass
        else:
            img = self.colour_transform(img)

        # normalizing
        img = normalize_img_tensor(img, imagenet=self.use_imagenet_norm)

        # assembling labelled dict
        labelled_dict = {}
        if np.random.rand() > 0.5:
            labelled_dict["box_A"] = get_random_crop_boxes(
                                                input_size=(self.crop_size, self.crop_size), 
                                                min_crop_ratio=self.min_crop_ratio, 
                                                max_crop_ratio=self.max_crop_ratio,
                                                )
        else:
            labelled_dict["box_A"] = self.identity_crop_box
        labelled_dict["img"] = img
        labelled_dict["label"] = label_img

        if not self.only_labelled:
            # reading in unlabelled BDD image
            img_path = self.bdd_examples[index]
            bdd_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
            bdd_img = self.preprocess_data(bdd_img, label=None)

            # creating two pixel-wise aligned imgs with different appearance
            if self.no_appearance_transform:
                bdd_img_1 = bdd_img
                bdd_img_2 = bdd_img
            else:
                bdd_img_1 = self.colour_transform(bdd_img)
                bdd_img_2 = self.colour_transform(bdd_img)

            # normalizing
            bdd_img_1 = normalize_img_tensor(bdd_img_1, imagenet=self.use_imagenet_norm)
            bdd_img_2 = normalize_img_tensor(bdd_img_2, imagenet=self.use_imagenet_norm)

            # assembling bdd dict
            bdd_dict = {}
            bdd_dict["img_1"] = bdd_img_1
            bdd_dict["img_2"] = bdd_img_2

            if np.random.rand() > 0.5:
                bdd_dict["box_A"] = get_random_crop_boxes(input_size=(self.crop_size, self.crop_size), 
                                                            min_crop_ratio=self.min_crop_ratio, max_crop_ratio=self.max_crop_ratio)
                bdd_dict["box_B"] = self.identity_crop_box
            else:
                bdd_dict["box_A"] = self.identity_crop_box
                bdd_dict["box_B"] = get_random_crop_boxes(input_size=(self.crop_size, self.crop_size), 
                                                            min_crop_ratio=self.min_crop_ratio, max_crop_ratio=self.max_crop_ratio)

            return (labelled_dict, bdd_dict)
        else:
            return labelled_dict, {}


    def __len__(self):
        return self.num_bdd_examples
    

if __name__ == "__main__":
    dataset = CityscapesxBDDDataset(
                        labelled_dataroot="/Users/dw/data",
                        raw_dataroot="/Users/dw/data/bdd100k",
                        sensor_models_path=None,
                        sax_domain=None,
                        add_resize_noise=False,
                        no_appearance_transform=True,
                        )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, (labelled_dict, raw_dict) in enumerate(dataloader):
        print(raw_dict["img_1"].shape)
        print(raw_dict["img_2"].shape)

        # denormalizing imgs
        raw_dict["img_1"] = raw_dict["img_1"] / 2 + 0.5
        raw_dict["img_2"] = raw_dict["img_2"] / 2 + 0.5

        # visualizing raw images
        img_1 = raw_dict["img_1"].squeeze().permute(1,2,0).numpy()
        img_2 = raw_dict["img_2"].squeeze().permute(1,2,0).numpy()
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
        cv2.imshow("img_1", img_1)
        cv2.imshow("img_2", img_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
