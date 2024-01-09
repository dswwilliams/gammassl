import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import sys
import copy
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
        from models.setr_decode_head import SETRUPHead_M2F
        decoder_pup = SETRUPHead_M2F(
                                global_scales=[0.5, 1, 2, 4],
                                in_channels=self.backbone_dim,
                                out_channels=self.intermediate_dim,
                                )
        
        from models.mask2former import MultiScaleMaskedTransformerDecoder
        transformer_decoder = MultiScaleMaskedTransformerDecoder(
                                                        in_channels=256,
                                                        mask_classification=True,       # has to be True
                                                        num_classes=19,                 # affects the linear layer producing the outputs_class
                                                        hidden_dim=256,                 # dimensionality of features throughout the transformer decoder (self-attention, cross-attention, mlp etc.)
                                                        num_queries=19,                 # number of query features
                                                        nheads=self.opt.nheads,                       # number of heads for self-attention and cross-attention
                                                        dim_feedforward=256,           # the internal dimensionality of the FFN (in FFN: hidden_dim -> dim_feedforward -> hidden_dim with Linear layers)
                                                        dec_layers=3,                   # number of rounds of [self-attn, cross-attn, FFN] to do
                                                        pre_norm=True,                  # whether not do layer norm at start or end of attn and FFN layers
                                                        mask_dim=256,                   # dim of mask embed, which needs to the same as the dim of the mask features
                                                        enforce_input_project=False,
                                                        )
        if self.opt.proj_query_pup:
            target_pup = decoder_pup

            from models.projection_networks import ProjectionMLP
            projection_net = ProjectionMLP(
                                    input_feature_len=self.backbone_dim,
                                    output_feature_len=self.backbone_dim, 
                                    dropout_prob=None,
                                    ).to(self.device)
            # project at low res -> learned upsampling
            query_pup = nn.Sequential(projection_net, copy.deepcopy(target_pup),)

            class Decoder(nn.Module):
                def __init__(self,):
                    super().__init__()
                    """
                    - have two different progressive upsamplers for each of target and query branches
                    """
                    self.target_pup = target_pup
                    self.query_pup = query_pup
                    self.transformer_decoder = transformer_decoder
                def forward():
                    pass

        else:
            class Decoder(nn.Module):
                def __init__(self,):
                    super().__init__()
                    self.decoder_pup = decoder_pup
                    self.transformer_decoder = transformer_decoder
                def forward():
                    pass

        self.decode_head = Decoder()

        ################################################################################################

        ################################################################################################
        ### define projection network ###
        if not self.opt.skip_projection:
            nonproj_len = self.intermediate_dim if self.opt.use_deep_features else self.backbone_dim
            from models.projection_networks import ProjectionMLP
            self.projection_net = ProjectionMLP(
                                    input_feature_len=nonproj_len, 
                                    output_feature_len=self.prototype_len, 
                                    dropout_prob=None,
                                    ).to(self.device)
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
            print("no grad for vit")
            with torch.no_grad():
                shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]

        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.backbone.patch_size, W//self.backbone.patch_size)

        if (self.opt.use_deep_features) or (use_deep_features):
            deep_features = self.decode_features(shallow_features, return_list=False)
            return deep_features
        else:
            return shallow_features
        
    def extract_m2f_output(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        - should be similar to target branch rather than query branch
        """

        H, W = x.shape[-2:]
        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad for vit")
            with torch.no_grad():
                shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.backbone.patch_size, W//self.backbone.patch_size)


        # do this with the target branch, extract_m2f_output() is used in labelled task
        deep_features_list = self.decode_features(shallow_features, return_list=True, target=True)

        output = self.decode_head.transformer_decoder(x=deep_features_list[:-1], mask_features=deep_features_list[-1])
        output["pred_masks"] = F.interpolate(output["pred_masks"], size=(H, W), mode="bilinear", align_corners=False)
        return output

    
    def decode_features(self, x, return_list=True, target=False, query=False):
        """
        decodes features from backbone
        input: x.shape = [batch_size, (H//patch_size)*(W//patch_size)+1, backbone_dim]
        """
        if target and self.opt.proj_query_pup:
            if return_list:
                return self.decode_head.target_pup(x)
            else:
                return self.decode_head.target_pup(x)[-1]
        elif query and self.opt.proj_query_pup:
            if return_list:
                return self.decode_head.query_pup(x)
            else:
                return self.decode_head.query_pup(x)[-1]
        else:
            if return_list:
                return self.decode_head.decoder_pup(x)
            else:
                return self.decode_head.decoder_pup(x)[-1]

    def extract_proj_features(self, x, masks=None): 
        """ extract projected features """
        features = self.extract_features(x, masks=masks)
        proj_features = self.projection_net(features)
        return proj_features
    
    @staticmethod
    def semantic_inference(mask_cls, mask_pred, use_sigmoid=False):

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        if use_sigmoid:
            mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def get_seg_masks(self, x, include_void=False, high_res=False, masks=None, target=False, query=False, return_mask_features=False, use_sigmoid=False):
        H, W = x.shape[-2:]
        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad for vit")
            with torch.no_grad():
                shallow_features = self.backbone.forward_features(x, masks=masks)["x_norm_patchtokens"]
        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.backbone.patch_size, W//self.backbone.patch_size)


        deep_features_list = self.decode_features(shallow_features, return_list=True, target=target, query=query)

        output = self.decode_head.transformer_decoder(x=deep_features_list[:-1], mask_features=deep_features_list[-1])
        if high_res:
            output["pred_masks"] = F.interpolate(output["pred_masks"], size=(H, W), mode="bilinear", align_corners=False)

        seg_masks = self.semantic_inference(output["pred_logits"], output["pred_masks"], use_sigmoid=use_sigmoid)

        if return_mask_features:
            return seg_masks, deep_features_list[-1]
        else:
            return seg_masks

    def get_target_seg_masks(self, x, include_void=False, high_res=False, masks=None, return_mask_features=False, use_sigmoid=False):
        return self.get_seg_masks(x, include_void=include_void, high_res=high_res, masks=masks, target=True, return_mask_features=return_mask_features, use_sigmoid=use_sigmoid)

    def get_query_seg_masks(self, x, include_void=False, high_res=False, masks=None, return_mask_features=False, use_sigmoid=False):
        return self.get_seg_masks(x, include_void=include_void, high_res=high_res, masks=masks, query=True, return_mask_features=return_mask_features, use_sigmoid=use_sigmoid)

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
    argparser.add_argument("--nheads", type=int, default=1)
    opt = argparser.parse_args()

    seg_net = ViTDINOSegNet(device="cpu", opt=opt, num_known_classes=19)
    seg_net.eval()

    x = torch.randn(1, 3, 224, 224)

    output = seg_net.extract_m2f_output(x)

    print("pred_logits", output["pred_logits"].shape)
    print("pred_masks", output["pred_masks"].shape)


    seg_masks = seg_net.get_seg_masks(x, include_void=False, high_res=True)
    print(seg_masks.shape)


    # for f in output:
    #     print(f.shape)

    # output = seg_net.transformer_decoder(x=output[:-1], mask_features=output[-1])
    
    # print("pred_logits", output["pred_logits"].shape)
    # print("pred_masks", output["pred_masks"].shape)

    # x = torch.randn(1, 3, 224, 224)
    # output = seg_net.extract_features(x)
    # print(output.shape)
    # output = seg_net.extract_proj_features(x)
    # print(output.shape)
    # output = seg_net.get_seg_masks(x, include_void=False, high_res=True)
    # print(output.shape)


    # for key in seg_net.state_dict().keys():
    #     if "lora" in key:
    #         print(key)