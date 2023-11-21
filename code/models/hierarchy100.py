from micromind import MicroMind
from micromind.networks import PhiNet

import torch
import torch.nn as nn
import numpy as np

model_path = "./pretrained/hierarchy10/epoch_48_val_loss_0.6899.ckpt"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
elif torch.backends.mps.is_available: 
    device = torch.device("mps")
    print("Running on the MPS")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

clustering_mapping = {1: 0, 4: 0, 30: 0, 32: 0, 55: 0, 67: 0, 72: 0, 73: 0, 91: 0, 95: 0, 47: 1, 52: 1, 54: 1, 56: 1, 59: 1, 62: 1, 70: 1, 82: 1, 92: 1, 96: 1, 0: 2, 9: 2, 10: 2, 16: 2, 28: 2, 51: 2, 53: 2, 57: 2, 61: 2, 83: 2, 5: 3, 20: 3, 22: 3, 25: 3, 39: 3, 40: 3, 84: 3, 86: 3, 87: 3, 94: 3, 6: 4, 7: 4, 14: 4, 18: 4, 24: 4, 26: 4, 45: 4, 77: 4, 79: 4, 99: 4, 12: 5, 17: 5, 23: 5, 33: 5, 37: 5, 49: 5, 60: 5, 68: 5, 71: 5, 76: 5, 3: 6, 15: 6, 19: 6, 21: 6, 31: 6, 38: 6, 42: 6, 43: 6, 88: 6, 97: 6, 34: 7, 36: 7, 50: 7, 63: 7, 64: 7, 65: 7, 66: 7, 74: 7, 75: 7, 80: 7, 8: 8, 13: 8, 41: 8, 48: 8, 58: 8, 69: 8, 81: 8, 85: 8, 89: 8, 90: 8, 2: 9, 11: 9, 27: 9, 29: 9, 35: 9, 44: 9, 46: 9, 78: 9, 93: 9, 98: 9}

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

        self.modifier_bias = torch.randn(self.output, self.input, requires_grad=True, device=device)
        #self.modifier_bias.requires_grad = True

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
        x = self.modules["classifier"](feature_vector)
        indices_1 = torch.argmax(x, dim=1)

        # indices_np = indices_1.to('cpu').numpy()
        # test = batch[1].to('cpu').numpy()
        # print("trut", test)
        # print("predicted", indices_1)
        # test = np.array([clustering_mapping[y] for y in test])
        # print("trut_cluster", test)
        # print("predicted", indices_np)
        # print(test == indices_np )
        # print((test == indices_np).sum()/len(indices_1))        
        # print(test)
        # print(indices_np)
        # print(test == indices_np)
        # print((test == indices_np).sum(0))
        # print(torch.tensor(indices_1.tolist() == test).sum()/len(indices_1))
        
        #broadcasted_bias = self.modifier_bias.unsqueeze(0).expand(len(batch[0]),-1,-1)
        #out_expanded = softmax.unsqueeze(2)
        #shifts = (out_expanded * broadcasted_bias).sum(dim=1) 

        shifts = torch.index_select(self.modifier_bias, 0, indices_1)
        shifted = feature_vector + shifts
        last = self.modules["classifier"](shifted)

        softmax2 = DiffSoftmax(last, tau=1.0, hard=False, dim=1)

        # Find the index of the 1
        #indices_of_ones1 = (softmax * self.indices).sum(dim=1)

        indices_of_ones2 = torch.argmax(last, dim=1)
        # print(indices_of_ones2)

        #indexes_int = torch.argmax(softmax, dim=1)

        #output_tensor = torch.cat([torch.zeros(10*indexes_int, batch[0].shape(0)), softmax2, torch.zeros(10*(10-indexes_int-1))], dim=0)
        #output_tensor = torch.stack([torch.cat([torch.zeros(10*indexes_int[x]), softmax2, torch.zeros(10*(10-indexes_int[x]-1))], dim=0) for x in range(n)], dim=0)

        # print(self.modifier_bias)

        # Calculate the ranges using vectorized operations
        start = indices_1 * 10
        end = (indices_1 + 1) * 10

        output_tensor = torch.zeros(len(batch[0]), 100, device=device)

        # Use 'torch.arange' and 'torch.stack' for creating the sequence tensors
        to_add = torch.stack([torch.arange(start[i], end[i], device=device) for i in range(len(indices_1))])

        output_tensor.scatter_(1, to_add, softmax2)        

        return output_tensor


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
            #print(list(self.modules.parameters()))
            #opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
            opt = torch.optim.Adam([self.modifier_bias], self.hparams.lr)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
            #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD([self.modifier_bias], self.hparams.lr)
        return opt, sched