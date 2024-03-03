import torch.nn as nn


class ProjectionMLP(nn.Module):
    """
    Pytorch module for projection multi-layer perceptron.
    Projects input features into a different feature space on a per-pixel basis.
    """
    def __init__(self, input_feature_len=128, output_feature_len=None, dropout_prob=None):
        """ 
        Args:
            input_feature_len (int): Dimensionality of input features.
            output_feature_len (int): Dimensionality of output features.
            dropout_prob (float): Dropout probability. Default: None.
        """        
        super(ProjectionMLP, self).__init__()

        if output_feature_len is None:
            self.output_feature_len = input_feature_len
        else:
            self.output_feature_len = output_feature_len
        
        # linear layers
        self.linear_1 = nn.Linear(in_features=input_feature_len, out_features=input_feature_len, bias=False)
        self.linear_2 = nn.Linear(in_features=input_feature_len, out_features=self.output_feature_len, bias=True)

        if dropout_prob is not None:
            self.regulariser = nn.Dropout(p=dropout_prob)
        else:
            self.regulariser = nn.BatchNorm1d(num_features=input_feature_len, affine=True)

        # activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: input features of shape = [batch_size, input_feature_len, H, W]

        Returns:
            projected_features: Projected features of shape = [batch_size, output_feature_len, H, W]
        """
        bs, feature_len, h, w = x.shape
        x = x.permute(0,2,3,1)
        x = x.reshape(bs*h*w, feature_len)
        x = self.relu(self.regulariser(self.linear_1(x)))
        x = self.linear_2(x)

        # semantic features for matching
        projected_features = x.reshape(bs, h, w, self.output_feature_len).permute(0,3,1,2)

        return projected_features
