from micromind import Metric
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchinfo import summary
from ptflops import get_model_complexity_info

from torchvision.transforms import v2
from torch.utils.data import default_collate

import dataset.dataset as dataset

import os
import random
import importlib
import numpy as np

batch_size = 32

def START_seed():
    seed = 9
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def save_parameters(model, hparams):

    path = hparams.output_folder + "/" + hparams.experiment_name

    input = (3, 32, 32)
    macs_backbone, params_backbone = get_model_complexity_info(model.modules["feature_extractor"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_backbone = summary(model.modules["feature_extractor"], input_size=(batch_size, 3, 32, 32))    
    #print(summary_backbone)

    input = (model.input, 1, 1)
    macs_classifier, params_classifier = get_model_complexity_info(model.modules["classifier"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_classifier = summary(model.modules["classifier"], input_size=(10, model.input, 1, 1))    

    output = "BACKBONE\n" 
    output += "MACs {}, learnable parameters {}\n".format(macs_backbone, params_backbone)
    output += str(summary_backbone) + "\n"
    output += "\n"*2
    output += "CLASSIFIER\n" 
    output += "MACs {}, learnable parameters {}\n".format(macs_classifier, params_classifier)
    output += str(summary_classifier)        

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/architecture.txt', 'w') as file:
        file.write(output)    

    with open(path + '/meta.txt', 'w') as file:
        file.write(str(hparams))

if __name__ == "__main__":  

    START_seed()  
    
    hparams = parse_arguments()    
    d = hparams.d
    hparams.experiment_name = hparams.experiment_name + '/' + str(hparams.d) + '/'   
    print(hparams.experiment_name)

    print("Running experiment with {}".format(hparams.d))

    module = importlib.import_module("models." + hparams.model_name)
    ImageClassification = getattr(module, "ImageClassification")     

    m = ImageClassification(hparams, inner_layer_width = hparams.d)

    def compute_accuracy(pred, batch): 
        if(len(batch[1].shape)==1):   
            tmp = (pred.argmax(1) == batch[1]).float()                                    
        else:
            tmp = (pred.argmax(1) == batch[1].argmax(1)).float()
        return tmp
    
    cutmix = v2.CutMix(num_classes=20, alpha=0.5)
    mixup = v2.MixUp(num_classes=20, alpha=0.5)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        res = cutmix_or_mixup(*default_collate(batch))        
        return [x for x in res]  
        
    train_transform = transforms.Compose(
        [
         transforms.ToTensor(), 
         transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)), 
         transforms.Resize((160, 160), antialias=True), 
         transforms.RandomHorizontalFlip(0.5),
         transforms.RandomRotation(10)
        ] 
    )
    transform = transforms.Compose(
        [
         transforms.ToTensor(), 
         transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)), 
         transforms.Resize((160, 160), antialias=True),          
        ] 
    )
    trainset = dataset.CIFAR100CUSTOM(
        root="data/cifar-100", train=True, download=True, transform=train_transform, coarse=True
    )
    testset = dataset.CIFAR100CUSTOM(
        root="data/cifar-100", train=False, download=True, transform=transform, coarse=True
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        #collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
    )

    acc = Metric(name="accuracy", fn=compute_accuracy)    

    epochs = hparams.epochs 

    m.train(
        epochs=epochs,
        datasets={"train": train_loader, "val": test_loader},
        metrics=[acc],
        debug=hparams.debug,
    )