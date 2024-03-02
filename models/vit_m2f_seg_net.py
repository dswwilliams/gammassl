import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from models.get_dino_from_repo import get_repo_dino
from models.base_seg_net import BaseSegNet
from utils.m2f_utils import semantic_inference
from models.setr_decode_head import SETRUPHead_M2F
from models.mask2former import MultiScaleMaskedTransformerDecoder

class ViT_M2F_Decoder(nn.Module):
    """
    Pytorch module for the decoder of the ViT_M2F_SegNet.
    Applies a progressively upsampling decoder, followed by a transformer decoder.
    """
    def __init__(self, decoder_pup, transformer_decoder):
        super().__init__()
        self.decoder_pup = decoder_pup
        self.transformer_decoder = transformer_decoder
    def forward(self, shallow_features):

        deep_features_list = self.decoder_pup(shallow_features)
        output = self.transformer_decoder(x=deep_features_list[:-1], mask_features=deep_features_list[-1])

        return output



class ViT_M2F_SegNet(BaseSegNet):
    """
    Pytorch module for a segmentation network that uses Vision Transformer (ViT) as the encoder and Mask2Former as the decoder.
    Based on: https://github.com/facebookresearch/Mask2Former
    """
    def __init__(self, device, opt, num_known_classes):
        super().__init__(device, opt, num_known_classes)
        self.opt = opt

        self.intermediate_dim = self.opt.intermediate_dim
        self.prototype_len = self.opt.prototype_len

        # defining encoder
        vit_dino = get_repo_dino(dino_path=self.opt.dino_path, lora_rank=self.opt.lora_rank, vit_size=self.opt.vit_size)
        self.encoder = vit_dino
        self.encoder_dim = vit_dino.embed_dim


        # decoder_pup Progressively UPsamples the features
        decoder_pup = SETRUPHead_M2F(
                                global_scales=[0.5, 1, 2, 4],
                                in_channels=self.encoder_dim,
                                out_channels=self.intermediate_dim,
                                )

        # transformer_decoder refines the initial query features conditioned on the pup features
        transformer_decoder = MultiScaleMaskedTransformerDecoder(
                                                        in_channels=256,
                                                        mask_classification=True,       # has to be True
                                                        num_classes=19,                 # affects the linear layer producing the outputs_class
                                                        hidden_dim=256,                 # dimensionality of features throughout the transformer decoder (self-attention, cross-attention, mlp etc.)
                                                        num_queries=19,                 # number of query features
                                                        nheads=self.opt.nheads,         # number of heads for self-attention and cross-attention
                                                        dim_feedforward=256,            # the internal dimensionality of the FFN (in FFN: hidden_dim -> dim_feedforward -> hidden_dim with Linear layers)
                                                        dec_layers=3,                   # number of rounds of [self-attn, cross-attn, FFN] to do
                                                        pre_norm=True,                  # whether not do layer norm at start or end of attn and FFN layers
                                                        mask_dim=256,                   # dim of mask embed, which needs to the same as the dim of the mask features
                                                        enforce_input_project=False,
                                                        )

        # defining overall decoder
        self.decoder = ViT_M2F_Decoder(decoder_pup, transformer_decoder)

        # defining projection network
        nonproj_len = self.intermediate_dim if self.opt.use_deep_features else self.encoder_dim
        from models.projection_networks import ProjectionMLP
        self.projection_net = ProjectionMLP(
                                input_feature_len=nonproj_len, 
                                output_feature_len=self.prototype_len, 
                                dropout_prob=None,
                                )

        # move model to device
        self.to(self.device)


    def extract_features(self, x: torch.Tensor, use_deep_features: bool = False, masks: torch.Tensor = None) -> torch.Tensor:
        """ 
        Extract high-dim features describing the input image x using the encoder, and optionally the decoder.

        Args: 
            x, image of shape = [batch_size, 3, H, W]

        Returns:
            deep_features, high-dim features of shape = [batch_size, (H//patch_size)*(W//patch_size), encoder_dim]
             if use_deep_features is True or self.opt.use_deep_features is True
             else, shallow_features, shape = [batch_size, (H//patch_size)*(W//patch_size), intermediate_dim]    
        """
        H, W = x.shape[-2:]

        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.encoder.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad")
            with torch.no_grad():
                shallow_features = self.encoder.forward_features(x, masks=masks)["x_norm_patchtokens"]

        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.encoder.patch_size, W//self.encoder.patch_size)

        if (self.opt.use_deep_features) or (use_deep_features):
            deep_features = self.decoder.decoder_pup(shallow_features)[-1]
            

            return deep_features
        else:
            return shallow_features
        
    def extract_m2f_output(self, x: torch.Tensor, masks: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Extracts the output of the Mask2Former network, given the input image x and optionally the masks.
        The m2f_output is a dictionary containing the predicted logits and masks, which can be combined to get the final segmentation masks.

        Args:
            x: input image of shape = [batch_size, 3, H, W]

        Returns:
            m2f_outout, dictionary of the form:
                m2f_output = {
                    "pred_logits": torch.Tensor, shape = [batch_size, num_classes, num_queries+1]   (see definition of transformer_decoder)
                    "pred_masks": torch.Tensor, shape = [batch_size, num_classes, H, W]
                }
        """

        H, W = x.shape[-2:]
        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.encoder.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad")
            with torch.no_grad():
                shallow_features = self.encoder.forward_features(x, masks=masks)["x_norm_patchtokens"]
        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.encoder.patch_size, W//self.encoder.patch_size)


        m2f_output = self.decoder(shallow_features)    
        m2f_output["pred_masks"] = F.interpolate(m2f_output["pred_masks"], size=(H, W), mode="bilinear", align_corners=False)

        return m2f_output

    def extract_proj_features(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts features from x, then projects them using the projection network.

        Args:
            x: input image of shape = [batch_size, 3, H, W]

        Returns:
            proj_features: projected features of shape = [batch_size, prototype_len, (H//patch_size), (W//patch_size)]
        """
        features = self.extract_features(x, masks=masks)
        proj_features = self.projection_net(features)
        return proj_features
    
    def get_seg_masks(self, x: torch.Tensor, high_res=False, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Get segmentation masks from the input image x.

        Args:
            x: input image of shape = [batch_size, 3, H, W]

        Returns:
            seg_masks: segmentation masks of shape = [batch_size, num_classes, H, W] if high_res is True
                else, seg_masks.shape = [batch_size, num_classes, (H//patch_size), (W//patch_size)] if high_res is False

        """
        H, W = x.shape[-2:]
        # we want grads if lora rank is not None or if train_vit is True
        if (self.opt.lora_rank is not None) or (self.opt.train_vit):
            shallow_features = self.encoder.forward_features(x, masks=masks)["x_norm_patchtokens"]
        else:
            print("no grad")
            with torch.no_grad():
                shallow_features = self.encoder.forward_features(x, masks=masks)["x_norm_patchtokens"]
        if len(shallow_features.shape) == 3:
            bs, _, C = shallow_features.shape
            shallow_features = shallow_features.permute(0, 2, 1)      # [bs, C, N]
            shallow_features = shallow_features.reshape(bs, C, H//self.encoder.patch_size, W//self.encoder.patch_size)
            
        m2f_output = self.decoder(shallow_features)

        if high_res:
            m2f_output["pred_masks"] = F.interpolate(m2f_output["pred_masks"], size=(H, W), mode="bilinear", align_corners=False)

        seg_masks = semantic_inference(m2f_output["pred_logits"], m2f_output["pred_masks"])

        return seg_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_seg_masks(x, include_void=False, high_res=True)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dino_path", type=str, default="/Users/dw/networks/dinov2/dinov2.pth")
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

    seg_net = ViT_M2F_SegNet(device="cpu", opt=opt, num_known_classes=19)
    seg_net.eval()

    x = torch.randn(1, 3, 224, 224)

    output = seg_net.extract_features(x)
    print(output.shape)

    output = seg_net.extract_proj_features(x)
    print(output.shape)

    output = seg_net.extract_m2f_output(x)

    print("pred_logits", output["pred_logits"].shape)
    print("pred_masks", output["pred_masks"].shape)


    seg_masks = seg_net.get_seg_masks(x, high_res=True)
    print(seg_masks.shape)