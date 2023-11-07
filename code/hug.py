from micromind import Metric
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import random
import numpy as np

from torchinfo import summary
from ptflops import get_model_complexity_info

from model_original_pre import ImageClassification

import importlib

import os

batch_size = 128

print("hey")

def START_seed():
    seed = 9
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def save_parameters(model, path):

    input = (3, 32, 32)
    macs_backbone, params_backbone = get_model_complexity_info(model.modules["feature_extractor"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_backbone = summary(model.modules["feature_extractor"], input_size=(batch_size, 3, 32, 32))    
    print(summary_backbone)

    input = (model.input, 1, 1)
    macs_classifier, params_classifier = get_model_complexity_info(model.modules["original_classifier"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_classifier = summary(model.modules["original_classifier"], input_size=(10, model.input, 1, 1))    

    output = "BACKBONE\n" 
    output += "MACs {}, learnable parameters {}\n".format(macs_backbone, params_backbone)
    output += str(summary_backbone) + "\n"
    output += "\n"*2
    output += "CLASSIFIER\n" 
    output += "MACs {}, learnable parameters {}\n".format(macs_classifier, params_classifier)
    output += str(summary_classifier)    

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'architecture.txt', 'w') as file:
        file.write(output)
    
if __name__ == "__main__":  

   # START_seed()  

    hparams = parse_arguments()  
    hparams.lr = 0.0001
    #hparams.output_folder = 'results/adaptive_exp/a' + alphas_str[alpha_id] + '/'+str(exp)+'/' + str(d) + '/'
    print("Running experiment with")       

    # module = importlib.import_module(hparams.model_name)
    # ImageClassification = getattr(module, "ImageClassification") 

    # g = torch.Generator()
    # g.manual_seed(0)

    m = ImageClassification(hparams)

    def compute_accuracy(pred, batch):
        tmp = (pred.argmax(1) == batch[1]).float()
        return tmp
    
    ## datasets loads
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((160, 160), antialias=True)] 
    )
    trainset = torchvision.datasets.CIFAR100(
        root="data/cifar-100", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="data/cifar-100", train=False, download=True, transform=transform
    )
    
    ## split into train, val, test 
    val_size = int(0.1 * len(trainset))
    train_size = len(trainset) - val_size
    train, val = torch.utils.data.random_split(trainset, [train_size, val_size])    

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 

    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 

    )    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, 
        shuffle=False, 
        num_workers=1, 

    )

    print("Trainset size: ", len(train)//batch_size)
    print("Valset size: ", len(val)//batch_size)
    print("Testset size: ", len(testset)//batch_size)

    # save_parameters(m, hparams.output_folder)    

    acc = Metric(name="accuracy", fn=compute_accuracy)

    epochs = 200

    m.train(
        epochs=epochs,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[acc],
        debug=hparams.debug,
    )

    result = m.test(
        datasets={"test": test_loader},
    )    

    result += " Epochs: " + str(epochs)

    with open(hparams.output_folder + 'test_set_result.txt', 'w') as file:
        file.write(result)


### TODO

# train from scratch
