import torch as T
import torch.nn as nn
import kornia
import random


class ImgColourTransform(nn.Module):
    def __init__(self, n_seq_transforms, no_colour):
        super(ImgColourTransform, self).__init__()
        self.k = n_seq_transforms
        # corruption transforms
        motion_blur = kornia.augmentation.RandomMotionBlur(kernel_size=(3,3), angle=(-30,30),  direction=(-1, 1), p=1.)
        gaussian_noise = kornia.augmentation.RandomGaussianNoise(mean=0, std=0.01, p=1)
        # intensity transforms
        random_equalize = kornia.augmentation.RandomEqualize(p=1)
        sharpen = kornia.augmentation.RandomSharpness(1, p=1)
        # solarize = kornia.augmentation.RandomSolarize(0.1, 0.1, p=1)
        posterize = kornia.augmentation.RandomPosterize(6, p=1)
        # invert = kornia.augmentation.RandomInvert(p=1)
        # colour space transforms
        grayscale = kornia.augmentation.RandomGrayscale(p=1)
        channel_shuffle = kornia.augmentation.RandomChannelShuffle(p=1)
        self.jitter = kornia.augmentation.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=1.)

        self.no_colour = no_colour

        if self.no_colour:
            self.transform_list = [sharpen, gaussian_noise]
        else:
            self.transform_list = [sharpen, posterize, random_equalize, motion_blur, grayscale, gaussian_noise, channel_shuffle]


    def forward(self, img):
        # choose transforms from list
        transforms = random.sample(self.transform_list, k=self.k)
        if self.no_colour:
            pass
        else:
            transforms.append(self.jitter)
        # img needs to be in range: [0,1]
        if (img > 2).any():
            img = img/255
        for transform in transforms:
            img = T.clamp(transform(img.float()).float(), 0, 1)
        return img.squeeze()
    
if __name__ == "__main__":
    colour_transformer = ImgColourTransform(n_seq_transforms=1)

    img = colour_transformer(T.rand(3, 224, 224))
    
