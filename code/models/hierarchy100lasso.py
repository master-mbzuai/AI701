from micromind import MicroMind
from micromind.networks import PhiNet

import torch
import torch.nn as nn
import numpy as np

import platform

print(platform.system())

if platform.system() == "Darwin":
    model_path = "./code/pretrained/hierarchy10_new/epoch_52_val_loss_0.6459.ckpt"
else:
    model_path = "./pretrained/hierarchy10_new/epoch_52_val_loss_0.6459.ckpt"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
elif torch.backends.mps.is_available: 
    device = torch.device("mps")
    print("Running on the MPS")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

clustering_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 3, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5, 56: 5, 57: 5, 58: 5, 59: 5, 60: 6, 61: 6, 62: 6, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 68: 6, 69: 6, 70: 7, 71: 7, 72: 7, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7, 79: 7, 80: 8, 81: 8, 82: 8, 83: 8, 84: 8, 85: 8, 86: 8, 87: 8, 88: 8, 89: 8, 90: 9, 91: 9, 92: 9, 93: 9, 94: 9, 95: 9, 96: 9, 97: 9, 98: 9, 99: 9}


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class ImageClassification(MicroMind):

    # test 1 with n as input vector size and m classes custom d
    # n has to be calculated from the output of the neural network of the feature extractor

    def __init__(self, *args, inner_layer_width = 10, **kwargs):
        super().__init__(*args, **kwargs)

        self.input = 344
        self.output = 10

        self.modifier_weights = torch.randn(self.output, self.input, requires_grad=True, device=device)
        #self.modifier_bias = torch.randn(self.input, self.input, requires_grad=True, device=device)
        #self.modifier_bias.requires_grad = True

        self.lasso = nn.Parameter(torch.rand(self.input), requires_grad=True).to(device)
        self.lasso.requires_grad_ = True

        # alpha: 0.9
        # beta: 0.5
        # num_classes: 1000
        # num_layers: 7
        # t_zero: 4.0

        self.modules["feature_extractor"] = PhiNet(
            input_shape=(3, 240, 240),
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

        # Taking away the classifier from pretrained model
        pretrained_dict = torch.load(model_path, map_location=device)

        #loading the new model
        self.modules["feature_extractor"].load_state_dict(pretrained_dict["feature_extractor"])        
        for _, param in self.modules["feature_extractor"].named_parameters():
            param.requires_grad = False

        self.modules["flattener"] = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.modules["classifier"] = nn.Sequential(
            nn.Linear(in_features=self.input, out_features=self.output)    
        )
        self.modules["classifier"].load_state_dict(pretrained_dict["classifier"])
        for _, param in self.modules["classifier"].named_parameters():    
            param.requires_grad = False

    def forward(self, batch):

        feature_vector = self.modules["feature_extractor"](batch[0])
        feature_vector = self.modules["flattener"](feature_vector)
        feature_vector = torch.mul(feature_vector, self.lasso) # we need to add this as a computation complexity

        x = self.modules["classifier"](feature_vector)


        softmax1 = torch.argmax(x, dim=1)

        #feature_vector = feature_vector.reshape(len(batch[0]), 1, 344)

        weights = torch.index_select(self.modifier_weights, 0, softmax1)  

        shifted = torch.mul(weights, feature_vector)

        #weights = weights.view(344, 10, len(batch[0])).permute(2, 0, 1)
        #print(weights)

        last = self.modules["classifier"](shifted)
        softmax2 = DiffSoftmax(last, tau=1.0, hard=False, dim=1)

        # Calculate the ranges using vectorized operations
        softmax1 = torch.argmax(x, dim=1)
        start = softmax1 * 10
        end = (softmax1 + 1) * 10

        output_tensor = torch.zeros(len(batch[0]), 100, device=device)
        to_add = torch.stack([torch.arange(start[i], end[i], device=device) for i in range(len(softmax2))])
        return output_tensor.scatter_(1, to_add, softmax2)   


    def compute_loss(self, pred, batch):                
        lasso_loss = self.lasso.abs().sum() * self.alpha
        cross_loss = nn.CrossEntropyLoss()(pred, batch[1])
        return lasso_loss + cross_loss
    
    def configure_optimizers(self):
        """Configures and defines the optimizer for the task. Defaults to adam
        with lr=0.001; It can be overwritten by either passing arguments from the
        command line, or by overwriting this entire method.

        Returns
        ---------
            Optimizer and learning rate scheduler
            (not implemented yet). : Tuple[torch.optim.Adam, None]

        """

        assert self.hparams.opt in [
            "adam",
            "sgd",
        ], f"Optimizer {self.hparams.opt} not supported."
        if self.hparams.opt == "adam":
            #print(list(self.modules.parameters()))
            #opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
            opt = torch.optim.Adam([self.modifier_weights], self.hparams.lr)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
            #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD([self.modifier_bias], self.hparams.lr)
        return opt, sched