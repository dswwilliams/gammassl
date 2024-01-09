import torch
import torch.nn as nn
import math

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.norm_bias = False if bias else True
        self.norm = norm(num_features=out_channels, affine=self.norm_bias)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class SETRUPHead(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self,
                 total_upsample_factor=14,
                 kernel_size=3,
                 in_channels=256,
                 out_channels=256,
                 patch_size=None,
                 round_down_spatial_dims=True,
                 ):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(SETRUPHead, self).__init__()

        self.up_convs = nn.ModuleList()
        self.align_corners = False
        in_channels = in_channels
        out_channels = out_channels
        self.patch_size = patch_size

        default_up_scale = 2


        # min number of times to double spatial dims (another one would overshoot)
        min_num_ups = int(math.log2(total_upsample_factor) // math.log2(default_up_scale))


        scale_factors = min_num_ups * [2]
        if not round_down_spatial_dims:
            final_scale_factor = total_upsample_factor / (2 ** min_num_ups)
            scale_factors.append(final_scale_factor)


        for i in range(len(scale_factors)):
            self.up_convs.append(
                nn.Sequential(
                    # bundles conv norm activation
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2,
                        ),
                    Upsample(
                        scale_factor=scale_factors[i],
                        mode='bilinear',
                        align_corners=self.align_corners)))
            in_channels = out_channels


    def forward(self, x):
        for up_conv in self.up_convs:
            x = up_conv(x)
        return x

    def forward_features(self, x):
        outs = []
        for up_conv in self.up_convs:
            x = up_conv(x)
            outs.append(x)
        return outs


class SETRUPHead_M2F(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self,
                 global_scales=[0.5, 1, 2, 4],
                 kernel_size=3,
                 in_channels=384,
                 out_channels=256,
                 patch_size=None,
                 round_down_spatial_dims=True,
                 ):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super(SETRUPHead_M2F, self).__init__()

        self.up_convs = nn.ModuleList()
        self.align_corners = False
        in_channels = in_channels
        out_channels = out_channels
        self.patch_size = patch_size

        default_up_scale = 2

        downsample_scales = []
        upsample_scales = []
        for scale in global_scales:
            if scale < 1:
                downsample_scales.append(scale)
            else:
                upsample_scales.append(scale)
        

        relative_upsample_factors = []
        upsample_scales.insert(0, 1)
        for i in range(len(upsample_scales)-1):
            relative_upsample_factors.append(upsample_scales[i+1] / upsample_scales[i])
        print("relative_upsample_factors", relative_upsample_factors)


        relative_downsample_factors = []            # factors > 1 relate to downsampling, i.e. 2 means downsample by factor of 2
        downsample_scales.insert(0, 1)
        for i in range(len(downsample_scales)-1):
            relative_downsample_factors.append(int(downsample_scales[i] / downsample_scales[i+1]))
        print("relative_downsample_factors", relative_downsample_factors)
            

        up_in_channels = in_channels
        down_in_channels = in_channels
        for i in range(len(relative_upsample_factors)):
            self.up_convs.append(
                nn.Sequential(
                    # bundles conv norm activation
                    Upsample(
                        scale_factor=relative_upsample_factors[i],
                        mode='bilinear',
                        align_corners=self.align_corners),
                    ConvModule(
                        in_channels=up_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size - 1) // 2),
                        )
                        )
            up_in_channels = out_channels
        
        print(down_in_channels)
        self.down_pools = nn.ModuleList()
        for i in range(len(relative_downsample_factors)):
            self.down_pools.append(
                nn.Sequential(
                nn.AvgPool2d(kernel_size=relative_downsample_factors[i], stride=relative_downsample_factors[i]),
                ConvModule(
                    in_channels=down_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=int(kernel_size - 1) // 2),
                    )
                    )
            down_in_channels = out_channels


    def forward(self, x):
        outs = []
        down_x = x
        for down_pool in self.down_pools:
            down_x = down_pool(down_x)
            outs.append(down_x)
        up_x = x
        for up_conv in self.up_convs:
            up_x = up_conv(up_x)
            outs.append(up_x)
        return outs
    
if __name__ == "__main__":
    x = torch.randn(1, 384, 16, 16)
    model = SETRUPHead_M2F()

    y = model(x)

    for i in y:
        print(i.shape)