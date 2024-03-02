import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from models.base_seg_net import BaseSegNet
from segmentation_models_pytorch.encoders import get_encoder
# from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head
from segmentation_models_pytorch.base import (
    # SegmentationModel,
    SegmentationHead,
)
sys.path.append("../")
from models.deeplabv3_decoder import DeepLabV3PlusDecoder



class DeepLabDecoder(nn.Module):
    """
    Pytorch module for decoder of DeepLab segmentation model.
    """
    def __init__(self,decoder_aspp, segmentation_head):
        super().__init__()
        self.decoder_aspp = decoder_aspp
        self.segmentation_head = segmentation_head

    def forward(self, shallow_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            shallow_features: list of shallow features from the encoder,
                where shallow_features[-1] has shape = [batch_size, 512, (H//encoder_output_stride), (W//encoder_output_stride)]

        Returns:
            seg_masks: segmentation masks of shape = [batch_size, num_classes, (H//encoder_output_stride), (W//encoder_output_stride)]
        """
        deep_features = self.decoder_aspp(*shallow_features)
        seg_masks = self.segmentation_head(deep_features)
        return seg_masks


class DeepLabSegNet(BaseSegNet):
    """
    Pytorch module for DeepLab segmentation model.
    """
    def __init__(self, device, opt, num_known_classes,):
        super().__init__(device, opt, num_known_classes)

        self.opt = opt
        self.intermediate_dim = self.opt.intermediate_dim
        self.prototype_len = self.opt.prototype_len

        # defining encoder params
        in_channels = 3
        encoder_name = "resnet18"
        encoder_depth = 5
        encoder_weights =  "imagenet"
        encoder_output_stride = 8

        # defining decoder params
        decoder_channels = self.intermediate_dim
        decoder_atrous_rates = (12, 24, 36)
        
        # defining encoder 
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        # defining decoder
        decoder_aspp = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            )
        segmentation_head = SegmentationHead(
            in_channels=decoder_aspp.out_channels,
            out_channels=num_known_classes,
            kernel_size=1,
            upsampling=1,
            )
        self.decoder = DeepLabDecoder(decoder_aspp=decoder_aspp, segmentation_head=segmentation_head)

        # defining projection network
        nonproj_len = self.intermediate_dim if self.opt.use_deep_features else self.encoder.out_channels[-1]
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
            deep_features, high-dim features of shape = [batch_size, 256, (H//4), (W//4)]
             if use_deep_features is True or self.opt.use_deep_features is True
             else, shallow_features, shape = [batch_size, 512, (H//encoder_output_stride), (W//encoder_output_stride)]    
        """

        if self.opt.train_encoder:
            shallow_features = self.encoder(x)
        else:
            with torch.no_grad():
                shallow_features = self.encoder(x)

        if (self.opt.use_deep_features) or (use_deep_features):
            deep_features = self.decoder.decoder_aspp(*shallow_features)
            return deep_features
        else:
            return shallow_features[-1]

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
        decoder_output = self.extract_features(x, use_deep_features=True)
        masks = self.decoder.segmentation_head(decoder_output)
        if high_res:
            masks = F.interpolate(masks, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
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
    argparser.add_argument("--use_proto_seg", type=bool, default=False)
    argparser.add_argument("--train_encoder", type=bool, default=True)
    opt = argparser.parse_args()

    seg_net = DeepLabSegNet(device="cpu", opt=opt, num_known_classes=19)
    seg_net.eval()

    x = torch.randn(1, 3, 256, 256)

    output = seg_net.extract_features(x, use_deep_features=False)
    print(output.shape) 

    output = seg_net.extract_proj_features(x)
    print(output.shape)

    output = seg_net.get_seg_masks(x, high_res=True)
    print(output.shape)





