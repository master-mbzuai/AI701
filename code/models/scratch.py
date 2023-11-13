from micromind import MicroMind
from micromind.networks import PhiNet

import torch.nn as nn

class ImageClassification(MicroMind):

    # test 1 with n as input vector size and m classes custom d
    # n has to be calculated from the output of the neural network of the feature extractor
    def __init__(self, *args, inner_layer_width = 10, **kwargs):
        super().__init__(*args, **kwargs)

        self.input = 344
        self.output = 100        

        self.modules["feature_extractor"] = PhiNet(
            input_shape=(3, 224, 224),
            alpha=0.9,
            num_layers=7,
            beta=0.5,
            t_zero=4.0,
            include_top=False,
            num_classes=100,
            compatibility=False,
            divisor=8,
            downsampling_layers=[4,5,7]
        )        

    def forward(self, batch):
        x = self.modules["feature_extractor"](batch[0])           
        return x

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])    
