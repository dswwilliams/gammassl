
### Code for the projects presented in the following:

[“Mitigating Distributional Shift in Semantic Segmentation via Uncertainty Estimation from Unlabelled Data”, D. Williams, D. De Martini, M. Gadd, and P. Newman, IEEE Transactions on Robotics (T-RO), 2024](https://arxiv.org/abs/2402.17653)
<!-- TODO: link this to my personal website not arxiv -->
```
@article{gammassl,
title={{Mitigating Distributional Shift in Semantic Segmentation via Uncertainty Estimation from Unlabelled Data}},
author={Williams, David and De Martini, Daniele and Gadd, Matthew and Newman, Paul},
booktitle={IEEE Transactions on Robotics (T-RO)},
year={2024},
}
```

[“Masked Gamma-SSL: Learning Uncertainty Estimation via Masked Image Modeling”, D. Williams, M. Gadd, P. Newman, and D. De Martini, IEEE International Conference on Robotics and Automation (ICRA), 2024](https://arxiv.org/abs/2402.17622)
<!-- TODO: link this to my personal website not arxiv -->
```
@article{maskedgammassl,
title={{Masked Gamma-SSL: Learning Uncertainty Estimation via Masked Image Modeling}},
author={Williams, David and Gadd, Matthew and Newman, Paul and De Martini, Daniele},
booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
year={2024},
}
```

### Environment
The required conda environment can be setup with:
```
conda env create -f environment.yml
conda activate gammassl
```


### What should this repo do?
 * Implement both a ViT and a DeepLab segmentation network
 * Have two modes of training
    * Train from a frozen target
    * Train from an unfrozen target with prototype segmentation and the additional losses
* Should we make training flexible or just have a flag to change which training mode we are in?
    * Not sure what benefit flexibility brings really
    * At least for now maybe its just better to have a single flag: "training_mode"
* We also have the different input augmentations to contend with, but arguably they are independent of which training mode we are considering
* So he have 3 variables, each with two options
* But not all of the 9 options are things we have tested, but thats fine, can make that clear here in the README
* The key ones are:
    1. DeepLab + ProtoSeg with Additional Losses + Crop and Resize
    2. ViT + FrozenTarget + Crop and Resize
    3. ViT + FrozenTarget + Masking

* Model Testing in separate repo, or the same repo?
    * It's tempting to have a separate repo, but maybe its best to have it in both




