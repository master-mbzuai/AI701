from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import os
import torch
import torch.nn as nn

model_path = "./pretrained/finetuned/baseline.ckpt"

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

        self.input = 344
        self.output = 10

        # it is to be the number of decision points in the tree
        self.modifier_bias = nn.Parameter(torch.randn(self.output, self.input)).to(device)        

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
                nn.ReLU(),
                nn.Linear(in_features=self.input, out_features=self.output)      
        )

    def get_int_parameters(self, i, softmax_index):
        return (2 * i) + softmax_index
    
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


    def forward(self, batch):        
        feature_vector = self.modules["feature_extractor"](batch[0])                              
        x = self.modules["flattener"](feature_vector)

        # recurrent classifier that we need to learn
        # first step is to learn the first part of the classifier
        # what if instead that is also a learnable parameters over the different runs of the classifier

        n_layers = torch.ceil(torch.log2(torch.tensor(self.input))).int().item() -1
        # emtpty tensor of size n_layers, batch_size of datatype torch.bool
        choices = torch.zeros(n_layers, len(batch[0]), dtype=torch.bool, device=device)        

        for i in range(n_layers):
            x = self.modules["classifier"](x)
            softmax = self.DiffSoftmax(x, tau=1.0, hard=False, dim=-1)
            choices[i][:] = torch.bernoulli(softmax[:,1])
            shift_index = self.get_int_parameters(i, softmax)
            x = x + self.modifier_bias[shift_index]
            
        return x
       
    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])
    
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
            opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
            #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(self.modules.parameters(), self.hparams.lr)
        return opt, sched