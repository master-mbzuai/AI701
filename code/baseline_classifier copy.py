from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchinfo import summary

batch_size = 128

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input = 38
        self.output = 100        

        self.modules["feature_extractor"] = PhiNet(
             (3, 32, 32), include_top=True, num_classes=100
        )

        # Taking away the classifier from pretrained model
        pretrained_dict = torch.load("./pretrained/1000_epochs_baseline.ckpt", map_location=device)["feature_extractor"]                

        #loading the new model
        self.modules["feature_extractor"].load_state_dict(pretrained_dict)        

    def forward(self, batch):
        x = self.modules["feature_extractor"](batch[0])        
        return x

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])


if __name__ == "__main__":
    hparams = parse_arguments()
    m = ImageClassification(hparams)    

    def compute_accuracy(pred, batch):
        tmp = (pred.argmax(1) == batch[1]).float()
        return tmp

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="data/cifar-100", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    testset = torchvision.datasets.CIFAR100(
        root="data/cifar-100", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    acc = Metric(name="accuracy", fn=compute_accuracy)

    m.train(
        epochs=10,
        datasets={"train": trainloader, "val": testloader, "test": testloader},
        metrics=[acc],
        debug=hparams.debug,        
    )

    m.test(
        datasets={"test": testloader},
    )

    m.export("output_onnx", "onnx", (3, 32, 32))