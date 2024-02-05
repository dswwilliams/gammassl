import torch as T
import numpy as np
import os
import sys
import csv
import cv2
sys.path.append("../")
from utils.dataset_utils import get_initial_scaling_values, random_flip, normalize_img_tensor, ImgColourTransform, get_random_crop
from utils.candr_utils import get_random_crop_boxes

cityscapes_train_dirs = ["train/jena/", "train/zurich/", "train/weimar/", "train/ulm/", "train/tubingen/", "train/stuttgart/",
              "train/strasbourg/", "train/monchengladbach/", "train/krefeld/", "train/hanover/",
              "train/hamburg/", "train/erfurt/", "train/dusseldorf/", "train/darmstadt/", "train/cologne/",
              "train/bremen/", "train/bochum/", "train/aachen/"]

### definitions ###
DOWNSAMPLE_FACTOR = 2.7273
# CROP_SIZE = 208
RESIZE_NOISE_FACTOR = 1.5

def get_resize_noise(height, crop_size, noise_factor):
    """
    - we want to control ratio  = (height/crop_size)
    - we want noise range: [ratio - alpha , ratio + alpha]
    - so mean value is orig value = ratio
    - and also ratio > 1, to prevent bad crops

    """
    noise = 2 * (np.random.rand() - 0.5)        # range: [-1,1]
    ratio = (height/crop_size)        # how many times bigger is shorter side than crop_size 

    alpha = (ratio - 1) / noise_factor
    noise = ratio + noise * alpha         # range: [ratio - alpha , ratio + alpha]

    return noise
    

class CityscapesxBDDDataset(T.utils.data.Dataset):
    def __init__(self, labelled_dataroot, 
                        raw_dataroot, 
                        no_appearance_transform=False, 
                        min_crop_ratio=1.2, 
                        max_crop_ratio=3, 
                        add_resize_noise=True,
                        only_labelled=False,
                        use_imagenet_norm=True,
                        no_colour=False,
                        crop_size=208,
                        ):

        self.big_crop_size = crop_size
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
       
        ########################################################################################################################
        ### RAW DATA SETUP ###
        self.raw_examples = []
        dataset_root = os.path.join(raw_dataroot,"images", "100k")
        for dataset_type in os.listdir(dataset_root):
            if dataset_type in ["train"]:
                for img_name in os.listdir(os.path.join(dataset_root, dataset_type)):
                    if img_name[-4:] == ".jpg":
                        img_path = os.path.join(dataset_root, dataset_type, img_name)
                        self.raw_examples.append(img_path)
        self.num_raw_examples = len(self.raw_examples)
        ########################################################################################################################

        ########################################################################################################################
        ### CITYSCAPES DATA SETUP ###
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
        ########################################################################################################################

    def __getitem__(self, index):
        ########################################################################################################################
        ### getting labelled data ###
        example = np.random.choice(self.examples)

        img_path = example["img_path"]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
        
        # rescaling
        h_l, w_l,_ = img.shape
        h_l, w_l = get_initial_scaling_values(h_l, w_l, downsample_factor=self.downsample_factor)
        if self.add_resize_noise:
            # add noise to resize (as a proportion of how much bigger height is than crop_size)
            noise = get_resize_noise(h_l, self.big_crop_size, noise_factor=self.resize_noise_factor)
            w_l = int((w_l/h_l) * noise * self.big_crop_size)
            h_l = int(noise * self.big_crop_size)

        img = cv2.resize(img, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
        
        # flipping
        img, label_img = random_flip(img, label_img, p=0.5)
        # cropping
        img, label_img = get_random_crop(img, label_img, self.big_crop_size)        # randomness from where crop is, not over the scale
        # convert numpy -> torch:
        img = T.from_numpy(img).permute(2,0,1)/255
        label_img = T.from_numpy(label_img)
        
        # transforming appearance
        if self.no_appearance_transform:
            pass
        else:
            img = self.colour_transform(img)
        # normalizing
        img = normalize_img_tensor(img, imagenet=self.use_imagenet_norm)

        labelled_dict = {}
        if np.random.rand() > 0.5:
            labelled_dict["box_A"] = get_random_crop_boxes(
                                                input_size=(self.big_crop_size, self.big_crop_size), 
                                                min_crop_ratio=self.min_crop_ratio, 
                                                max_crop_ratio=self.max_crop_ratio,
                                                )
        else:
            labelled_dict["box_A"] = T.tensor([[0,0], [self.big_crop_size-1, 0], [self.big_crop_size-1, self.big_crop_size-1], [0, self.big_crop_size-1]])
        labelled_dict["img"] = img
        labelled_dict["label"] = label_img

        ########################################################################################################################
        if not self.only_labelled:
            ########################################################################################################################
            ### reading in raw_img ###
            raw_dict = {}
            img_path = self.raw_examples[index]
            raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
            # rescaling
            h_r, w_r,_ = raw_img.shape
            h_r, w_r = get_initial_scaling_values(h_r, w_r, downsample_factor=self.downsample_factor)
            if self.add_resize_noise:
                # add noise to resize (as a proportion of how much bigger height is than crop_size)
                noise = get_resize_noise(h_r, self.big_crop_size, noise_factor=self.resize_noise_factor)
                w_r = int((w_r/h_r) * noise * self.big_crop_size)
                h_r = int(noise * self.big_crop_size)

            raw_img = cv2.resize(raw_img, (w_r, h_r), interpolation=cv2.INTER_NEAREST)


            # cropping
            raw_img = get_random_crop(raw_img, crop_size=self.big_crop_size)

            # flipping 
            if np.random.rand() > 0.5:
                raw_img = random_flip(raw_img, p=1)

            # convert numpy -> torch:
            raw_img = T.from_numpy(raw_img).permute(2,0,1)
            raw_img = raw_img / 255

            # transforming appearance
            if self.no_appearance_transform:
                raw_img_1 = raw_img
                raw_img_2 = raw_img
            else:
                raw_img_1 = self.colour_transform(raw_img)
                raw_img_2 = self.colour_transform(raw_img)

            # normalizing
            raw_img_1 = normalize_img_tensor(raw_img_1, imagenet=self.use_imagenet_norm)
            raw_img_2 = normalize_img_tensor(raw_img_2, imagenet=self.use_imagenet_norm)

            raw_dict["img_1"] = raw_img_1
            raw_dict["img_2"] = raw_img_2

            if np.random.rand() > 0.5:
                raw_dict["box_A"] = get_random_crop_boxes(input_size=(self.big_crop_size, self.big_crop_size), 
                                                            min_crop_ratio=self.min_crop_ratio, max_crop_ratio=self.max_crop_ratio)
                raw_dict["box_B"] = T.tensor([[0,0], [self.big_crop_size-1, 0], [self.big_crop_size-1, self.big_crop_size-1], [0, self.big_crop_size-1]])
            else:
                raw_dict["box_A"] = T.tensor([[0,0], [self.big_crop_size-1, 0], [self.big_crop_size-1, self.big_crop_size-1], [0, self.big_crop_size-1]])
                raw_dict["box_B"] = get_random_crop_boxes(input_size=(self.big_crop_size, self.big_crop_size), 
                                                            min_crop_ratio=self.min_crop_ratio, max_crop_ratio=self.max_crop_ratio)
            ########################################################################################################################

            return (labelled_dict, raw_dict)
        else:
            return labelled_dict, {}


    def __len__(self):
        return self.num_raw_examples
    


if __name__ == "__main__":
    dataset = CityscapesxBDDDataset(
                        labelled_dataroot="/Users/dw/data",
                        raw_dataroot="/Users/dw/data/bdd100k",
                        sensor_models_path=None,
                        sax_domain=None,
                        add_resize_noise=False,
                        no_appearance_transform=True,
                        )
    
    dataloader = T.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

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
