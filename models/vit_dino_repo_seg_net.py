import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import sys
# sys.path.append("/Users/dw/code/pytorch/dinov2/dinov2")
# sys.path.append("/Users/dw/code/pytorch/dinov2")
sys.path.append("../")
from models.get_dino_from_repo import get_repo_dino
from models.base_seg_net import BaseSegNet

class ViTDINOSegNet(BaseSegNet):
    def __init__(self, device, opt, num_known_classes):
        super().__init__(device, opt, num_known_classes)
        self.opt = opt

        self.intermediate_dim = self.opt.intermediate_dim
        self.prototype_len = self.opt.prototype_len


        ################################################################################################
        ### defining backbone ###
        vit_dino = get_repo_dino(dino_path=self.opt.dino_path, dino_repo_path=self.opt.dino_repo_path, lora_rank=self.opt.lora_rank, vit_size=self.opt.vit_size)
        self.backbone = vit_dino
        self.backbone_dim = vit_dino.embed_dim
        ################################################################################################



        ################################################################################################
        ### choosing decoder ###

        # setr_pup
        from models.setr_decode_head import SETRUPHead
        self.decode_head = SETRUPHead(
                                total_upsample_factor=self.backbone.patch_size,
                                kernel_size=3, 
                                in_channels=self.backbone_dim, 
                                out_channels=self.intermediate_dim, 
                                patch_size=self.backbone.patch_size,
                                round_down_spatial_dims=True,
                                )
        ################################################################################################

        ################################################################################################
        ### define projection network ###
        nonproj_len = self.intermediate_dim if self.opt.use_deep_features else self.backbone_dim
        from models.projection_networks import ProjectionMLP
        # NOTE: just make output_feature_len = input_feature_len
        self.projection_net = ProjectionMLP(input_feature_len=nonproj_len, output_feature_len=self.prototype_len, dropout_prob=None).to(self.device)
        ################################################################################################
        
        ################################################################################################
        ### defining segmentation head ###
        self.seg_head = nn.Conv2d(in_channels=self.intermediate_dim, out_channels=self.num_output_classes, kernel_size=1)
        ################################################################################################


        ################################################################################################
        ### to device ###
        self.to_device()
        ################################################################################################


    def extract_features(self, x: torch.Tensor, use_deep_features: bool = False, masks: torch.Tensor = None) -> torch.Tensor:
        """ 
        extract features from backbone 
        input: x.shape = (batch_size, 3, H, W)

        if use_deep_features is True or self.opt.use_deep_features is True:
        output: features.shape = [batch_size, (H//patch_size)*(W//patch_size), backbone_dim]

        else:
        output: features.shape = [batch_size, (H//patch_size)*(W//patch_size), intermediate_dim]
    
        """
        H, W = x.shape[-2:]

        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad")
            with torch.no_grad():
                shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]


        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.backbone.patch_size, W//self.backbone.patch_size)

        if (self.opt.use_deep_features) or (use_deep_features):
            deep_features = self.decode_features(shallow_features)
            return deep_features
        else:
            return shallow_features
    """
    def extract_masked_features(self, x: torch.Tensor, use_deep_features: bool = False, masks: torch.Tensor = None) -> torch.Tensor:

        H, W = x.shape[-2:]

        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad")
            with torch.no_grad():
                shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]


        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.backbone.patch_size, W//self.backbone.patch_size)

        if (self.opt.use_deep_features) or (use_deep_features):
            deep_features = self.decode_features(shallow_features)
            return deep_features
        else:
            return shallow_features
    """

    """ 
    def extract_masked_features(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:

    
        H, W = x.shape[-2:]

        shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        bs, _, C = shallow_features.shape
        shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
        shallow_features = shallow_features.reshape(bs, C, H//self.backbone.patch_size, W//self.backbone.patch_size)

        return shallow_features
    """
    
    def decode_features(self, x):
        """
        decodes features from backbone
        input: x.shape = [batch_size, (H//patch_size)*(W//patch_size)+1, backbone_dim]
        """
        return self.decode_head(x)

    def extract_proj_features(self, x, masks=None): 
        """ extract projected features """
        features = self.extract_features(x, masks=masks)
        proj_features = self.projection_net(features)
        return proj_features

    def get_seg_masks(self, x, include_void=False, high_res=False):
        features = self.extract_features(x, use_deep_features=True)     # force decoding 
        seg_masks = self.seg_head(features)
        if not include_void:
            seg_masks = seg_masks[:,:self.num_known_classes,:,:]
        if high_res:
            return F.interpolate(seg_masks, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            return seg_masks

    def forward(self, x):
        return self.get_seg_masks(x, include_void=False, high_res=True)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dino_path", type=str, default="/Users/dw/networks/dinov2/dinov2.pth")
    argparser.add_argument("--dino_repo_path", type=str, default="/Users/dw/code/pytorch/dinov2")
    argparser.add_argument("--use_lora", type=bool, default=False)
    argparser.add_argument("--lora_rank", type=bool, default=4)
    argparser.add_argument("--include_void", type=bool, default=False)
    argparser.add_argument("--use_deep_features", type=bool, default=False)
    argparser.add_argument("--intermediate_dim", type=int, default=256)
    argparser.add_argument("--prototype_len", type=int, default=256)
    argparser.add_argument("--decode_head", type=str, default="setr")
    argparser.add_argument("--vit_size", type=str, default="small")
    opt = argparser.parse_args()

    seg_net = ViTDINOSegNet(device="cpu", opt=opt)
    seg_net.eval()

    x = torch.randn(1, 3, 224, 224)

    output = seg_net.extract_features(x)
    print(output.shape)
    output = seg_net.extract_proj_features(x)
    print(output.shape)
    output = seg_net.get_seg_masks(x, include_void=False, high_res=True)
    print(output.shape)


    for key in seg_net.state_dict().keys():
        if "lora" in key:
            print(key)