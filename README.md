# What does this repo do?
Implements methods for training semantic segmentation networks to perform high-quality uncertainty estimation on distributionally-shifted data. It does this by training on unlabelled distributionally-shifted images with a self-supervised task, and learning to detect segmentation inconsistency as a proxy for segmentation error.

`./scripts/train_deeplab_gssl.sh` represents our first method to do this in [GammaSSL](https://dswwilliams.github.io/posts/gammassl), and then `./scripts/train_vit_mgssl.sh` represents our second method [MaskedGammaSSL](https://dswwilliams.github.io/posts/mgssl).

Testing the quality of a model's uncertainty estimation can be performed using `ue_testing`.

The required conda environment can be setup with:
```
conda env create -f environment.yml
conda activate gammassl
```


## Papers

[“Mitigating Distributional Shift in Semantic Segmentation via Uncertainty Estimation from Unlabelled Data”, D. Williams, D. De Martini, M. Gadd, and P. Newman, IEEE Transactions on Robotics (T-RO), 2024](https://dswwilliams.github.io/posts/gammassl)

```
@article{gammassl,
title={{Mitigating Distributional Shift in Semantic Segmentation via Uncertainty Estimation from Unlabelled Data}},
author={Williams, David and De Martini, Daniele and Gadd, Matthew and Newman, Paul},
booktitle={IEEE Transactions on Robotics (T-RO)},
year={2024},
}
```

[“Masked Gamma-SSL: Learning Uncertainty Estimation via Masked Image Modeling”, D. Williams, M. Gadd, P. Newman, and D. De Martini, IEEE International Conference on Robotics and Automation (ICRA), 2024](https://dswwilliams.github.io/posts/mgssl)


```
@article{maskedgammassl,
title={{Masked Gamma-SSL: Learning Uncertainty Estimation via Masked Image Modeling}},
author={Williams, David and Gadd, Matthew and Newman, Paul and De Martini, Daniele},
booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
year={2024},
}
```