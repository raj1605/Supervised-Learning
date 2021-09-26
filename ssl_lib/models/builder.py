import numpy as np

from .resnet2 import ResNet
from .shakenet import ShakeNet
from .cnn13 import CNN13

# WRN-n-k denotes a residual network that has a total number of convolutional layers
# n and a widening factor k
def gen_model(name, num_classes, img_size):
    scale =  int(np.ceil(np.log2(img_size)))
    if name == "wrn":
        return ResNet(num_classes, 32, scale, 4)
    elif name == "shake":
        return ShakeNet(num_classes, 32, scale, 4)
    elif name == "cnn13":
        return CNN13(num_classes, 32)
    else:
        raise NotImplementedError