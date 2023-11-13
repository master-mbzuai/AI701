import torch, time, sys, os, copy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('../')

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from termcolor import cprint

from micromind import Metric
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate
from torchinfo import summary
from ptflops import get_model_complexity_info

from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import torch.nn as nn

model_path = "./pretrained/finetuned/baseline.ckpt"

# Spawn a separate process for each copy of the model
# mp.set_start_method('spawn')  # must be not fork, but spawn
queue = mp.Queue()

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
            num_classes=1000,
            compatibility=False,
            divisor=8,
            downsampling_layers=[4,5,7]
        )

        self.modules["classifier"] = nn.Sequential(                
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),  
                nn.Linear(in_features=self.input, out_features=self.output),
        )    

    def forward(self, batch):
        x = self.modules["feature_extractor"](batch[0])  
        x = self.modules["classifier"](x)      
        return x

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])  


# Define a function to train a single copy of the model
def train_model(queue, DEVICE, hparams):
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    batch_size = 64

    trainset, testset = queue.get()

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, 
        shuffle=False, 
        num_workers=1,    
    )

    with torch.no_grad():

        # Set the device to the current process's device
        # with torch.no_grad():
        m = ImageClassification(hparams, inner_layer_width=hparams.d)
        
        # Taking away the classifier from pretrained model
        pretrained_dict = torch.load(model_path, map_location=DEVICE)  

        test = pretrained_dict["feature_extractor"]

        #loading the new model
        m.modules["feature_extractor"].load_state_dict(test)        
        for _, param in m.modules["feature_extractor"].named_parameters():                
            param.requires_grad = False  

    print("Running experiment with {}".format(hparams.d))    

    def compute_accuracy(pred, batch):
        tmp = (pred.argmax(1) == batch[1]).float()
        return tmp

    acc = Metric(name="accuracy", fn=compute_accuracy)    

    epochs = hparams.epochs

    m.train(
        epochs=epochs,
        datasets={"train": train_loader, "val": test_loader},
        metrics=[acc],
        debug=hparams.debug,
    )   


NUM_MODEL_COPIES = 10
DEVICE = 'cuda:0'

hparams = parse_arguments()    

d = [10,25,50,75,90]
#d = [10]

processes = []
for rank in d:
    print("starting process with d = {}".format(rank))
    hparams.d = rank
    hparams.experiment_name = hparams.experiment_name + '/' + str(hparams.d) + '/'  
    process = mp.Process(target=train_model, args=(queue, DEVICE, hparams))
    process.start()
    processes.append(process)

time.sleep(2)

transform = transforms.Compose(
        [
         transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
         transforms.Resize((160, 160), antialias=True), 
         transforms.RandomHorizontalFlip(0.5),
         transforms.RandomRotation(10)
        ] 
    )

trainset = torchvision.datasets.CIFAR100(
    root="data/cifar-100", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR100(
    root="data/cifar-100", train=False, download=True, transform=transform
)



for rank in range(NUM_MODEL_COPIES):
    queue.put((trainset, testset))

# Wait for all processes to finish
for process in processes:
    process.join()