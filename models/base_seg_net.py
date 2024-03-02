import torch
import torch.nn as nn



class BaseSegNet(nn.Module):
    """ class that just deals with defining neural networks """
    def __init__(self, device, opt, num_known_classes):
        super().__init__()

        ################################################################################################
        ### generic to all ###
        self.opt = opt
        self.device = device

        self.num_known_classes = num_known_classes

        if self.opt.include_void:
            self.num_output_classes = self.num_known_classes + 1
        else:
            self.num_output_classes = self.num_known_classes
        ################################################################################################

        ################################################################################################
        self.encoder = None
        self.decoder = None
        ################################################################################################
        

    def extract_features(self, x, use_deep_features=False):
        return NotImplementedError
    
    def decode_features(self, x):
        return NotImplementedError

    def extract_proj_features(self, x): 
        return NotImplementedError

    def get_seg_masks(self, x, include_void=False, high_res=False):
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError