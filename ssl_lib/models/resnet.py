import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import leaky_relu, conv3x3, BatchNorm2d, param_init, BaseModel


class _Residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
    #consider two variants of residual block: add identity befor applying relue actviation or add
    #identity after activatation
        if activate_before_residual:
            self.pre_act = nn.Sequential(
                BatchNorm2d(input_channels),
                leaky_relu()
            )
        else:
            self.pre_act = nn.Identity()
        #the order of batch normalization, activation and convolution in residual block is BN-ReLU-conv
            layer.append(BatchNorm2d(input_channels))
            layer.append(leaky_relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(leaky_relu())
        layer.append(conv3x3(output_channels, output_channels))
      #adjust the dimenttion of the ouput of the residual block with the dimention of previous block (for adding)
        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        print('helooooo')
        x = self.pre_act(x)
        print("---------Inside forward ------------")
        return self.identity(x) + self.layer(x)


class ResNet(BaseModel):
    """
    ResNet
    Parameters
    --------
    num_classes: int
        number of classes
    filters: int
        number of filters
    scales: int
        number of scales
    repeat: int
        number of residual blocks per scale
    dropout: float
        dropout ratio (None indicates dropout is unused)
    """
    def __init__(self, num_classes, filters, scales, repeat, dropout=None, *args, **kwargs):
        super().__init__()
        feature_extractorr = [conv3x3(3, 16)]
        channels = 16
        for scale in range(scales):
            feature_extractorr.append(
                _Residual(channels, filters<<scale, 2 if scale else 1, activate_before_residual = (scale == 0))
            )
            channels = filters << scale
            for _ in range(repeat - 1):
                feature_extractorr.append(
                    _Residual(channels, channels)
                )

        feature_extractorr.append(BatchNorm2d(channels))
        feature_extractorr.append(leaky_relu())
        self.feature_extractor = nn.Sequential(*feature_extractorr)
        print(type(feature_extractorr), " : Type in old REsnet", feature_extractorr, " : Length in old resnet")
        classifier = []
        if dropout is not None:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(channels, num_classes))
        self.classifier = nn.Sequential(*classifier)

        param_init(self.modules())

    def forward(self, x):
        print("Inside Resnet's forward ------------------")