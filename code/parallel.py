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

from models.parallel import ImageClassification

from huggingface_hub import hf_hub_download

REPO_ID = "micromind/ImageNet"
FILENAME = "v7/state_dict.pth.tar"

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir="./pretrained")

# Spawn a separate process for each copy of the model
# mp.set_start_method('spawn')  # must be not fork, but spawn

queue = mp.Queue()


# Define a function to train a single copy of the model
def train_model(queue, DEVICE, hparams):
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    train_loader, val_loader, test_loader = queue.get()   

    # Set the device to the current process's device
    with torch.no_grad():
        model = ImageClassification(hparams, inner_layer_width=hparams.d).to(DEVICE)        

        # Taking away the classifier from pretrained model
        pretrained_dict = torch.load(model_path, map_location=DEVICE)
        model_dict = {}
        for k, v in pretrained_dict.items():
            if "classifier" not in k:
                model_dict[k] = v

        #loading the new model
        model.modules["feature_extractor"].load_state_dict(model_dict)
        for _, param in model.modules["feature_extractor"].named_parameters():
            param.requires_grad = False

        # if rank == 0:
        #     # changing weight in one model in a separate process doesn't affect the weights in the model in another process, because the weight tensors are not shared
        #     model.fc1.weight[0][0] = -33.0

        #     # but changing bias (which is a shared tensor) should affect biases in the other processes
        #     model.fc1.bias *= 4

        #     cprint(f'RANK: {rank} | {list(model.parameters())[0][0,0]}', color='magenta')

        # if rank == 8:
        #     cprint(f'RANK: {rank} | {list(model.parameters())[0][0,0]}', color='red')
        #     cprint(f'RANK: {rank} | BIAS: {model.fc1.bias}', color='red') 

    print("Running experiment with {}".format(hparams.d))    

    def compute_accuracy(pred, batch):
        tmp = (pred.argmax(1) == batch[1]).float()
        return tmp    

    print("Trainset size: ", len(train)//batch_size)
    print("Valset size: ", len(val)//batch_size)
    print("Testset size: ", len(testset)//batch_size)

    acc = Metric(name="accuracy", fn=compute_accuracy)    

    epochs = hparams.epochs

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


NUM_MODEL_COPIES = 10
DEVICE = 'cuda:0'

hparams = parse_arguments()    

d = [10,25,50,75,90]

processes = []
for rank in d:
    print("starting process with d = {}".format(rank))
    hparams.d = rank
    hparams.experiment_name = hparams.experiment_name + '/' + str(hparams.d) + '/'  
    process = mp.Process(target=train_model, args=(queue, DEVICE, hparams))
    process.start()
    processes.append(process)

time.sleep(2)

batch_size = 64

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

val_size = int(0.1 * len(trainset))
train_size = len(trainset) - val_size
train, val = torch.utils.data.random_split(trainset, [train_size, val_size])    

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, 
    shuffle=True, 
    num_workers=8, 
)
val_loader = torch.utils.data.DataLoader(
    val, batch_size=batch_size, 
    shuffle=False, 
    num_workers=8,    
)    
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, 
    shuffle=False, 
    num_workers=1,    
)

for rank in range(NUM_MODEL_COPIES):
    queue.put((train_loader, val_loader, test_loader))

# Wait for all processes to finish
for process in processes:
    process.join()