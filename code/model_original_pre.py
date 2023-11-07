from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "micromind/ImageNet"
FILENAME = "v5/state_dict.pth.tar"

model_path = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME), mmap_mode="rb"
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
elif torch.backends.mps.is_available: 
    device = torch.device("mps")
    print("Running on the MPS")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class ImageClassification(MicroMind):

    # test 1 with n as input vector size and m classes custom d
    # n has to be calculated from the output of the neural network of the feature extractor
    def __init__(self, *args, inner_layer_width = 10, **kwargs):
        super().__init__(*args, **kwargs)

        # alpha: 0.9
        # beta: 0.5
        # num_classes: 1000
        # num_layers: 7
        # t_zero: 4.0

        self.modules["feature_extractor"] = PhiNet(
            input_shape=(3, 224, 224),
            alpha=0.9,
            num_layers=7,
            beta=0.5,
            t_zero=4.0,
            include_top=True,
            num_classes=100,
            compatibility=False,
            divisor=8,
            downsampling_layers=[4,5,7]
        )

        print("evvai")

        #loading the new model
        self.modules["feature_extractor"].load_state_dict(torch.load(model_path))        

    def forward(self, batch):
        x = self.modules["feature_extractor"](batch[0])        
        return x

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])