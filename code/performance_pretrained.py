from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchinfo import summary

batch_size = 128

alpha = 3

class ImageClassification(MicroMind):

    # https://machine-learning.paperspace.com/wiki/accuracy-and-loss
    # test 1 with n as input vector size and m classes custom d
    # n has to be calculated from the output of the neural network of the feature extractor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["feature_extractor"] = PhiNet(
            (3, 32, 32), include_top=True, num_classes=100, alpha=alpha
        )        

    def forward(self, batch):
        return self.modules["feature_extractor"](batch[0])

    def compute_loss(self, pred, batch):
        #print(pred)
        return nn.CrossEntropyLoss()(pred, batch[1])


if __name__ == "__main__":
    hparams = parse_arguments()

    hparams.output_folder = 'pretrained/a' + str(alpha)
    hparams.lr = 0.01
    hparams.opt = 'adam'
    hparams.momentum = 0.03
    
    m = ImageClassification(hparams)

    summary(m.modules["feature_extractor"], input_size=(batch_size, 3, 32, 32))

    def compute_accuracy(pred, batch):
        tmp = (pred.argmax(1) == batch[1]).float()
        return tmp

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomResizedCrop(size=(32, 32), antialias=True), torchvision.transforms.RandomRotation(10)]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="data/cifar-100", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="data/cifar-100", train=False, download=True, transform=transform
    )

    ## split into train, val, test      
    val_size = 5000
    train_size = len(trainset) - val_size
    train, val = torch.utils.data.random_split(trainset, [train_size, val_size])    

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=1
    )    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    acc = Metric(name="accuracy", fn=compute_accuracy)

    m.train(
        epochs=50,
        datasets={"train": trainloader, "val":valloader},
        metrics=[acc],
        debug=hparams.debug,
    )

    result = m.test(
        datasets={"test": testloader},
    )

    with open(hparams.output_folder + 'test_set_result.txt', 'w') as file:
        file.write(result)