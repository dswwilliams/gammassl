

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


### TODO
* Link this repo to a forked version of the DINOv2 repo 
    * therefore can include my changes to it, e.g. lora
* Put up a forked version of DINOv2 repo on my github




