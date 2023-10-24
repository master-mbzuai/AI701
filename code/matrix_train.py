from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchinfo import summary

batch_size = 128

class ImageClassification(MicroMind):

    # test 1 with n as input vector size and m classes custom d
    # n has to be calculated from the output of the neural network of the feature extractor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input = 38
        self.output = 100
        self.d = 10

        self.modules["feature_extractor"] = PhiNet(
             #(3, 64, 64), include_top=False, alpha=1.335, beta=1.5
             (3, 32, 32), include_top=False
        )        

        # Taking away the classifier from pretrained model
        pretrained_dict = torch.load("./pretrained/phinet.ckpt")["classifier"]
        model_dict = {}
        for k, v in pretrained_dict.items():
            if "classifier" not in k:
                model_dict[k] = v

        #loading the new model
        self.modules["feature_extractor"].load_state_dict(model_dict)

        self.modules["classifier"] = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=self.input, out_features=self.output)
        )

        self.modules["adaptive_classifier"] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=self.input, out_features=self.d),
                # no relu
                nn.Linear(in_features=self.d, out_features=self.output)
            )        

    def forward(self, batch):
        x = self.modules["feature_extractor"](batch[0])
        #x = self.modules["classifier"](x)
        x = self.modules["adaptive_classifier"](x)
        return x

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])


if __name__ == "__main__":
    hparams = parse_arguments()
    m = ImageClassification(hparams)

    summary(m.modules["feature_extractor"], input_size=(batch_size, 3, 32, 32))
    summary(m.modules["classifier"], input_size=(128,m.input,1,1))
    summary(m.modules["adaptive_classifier"], input_size=(128,m.input,1,1))

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
        epochs=300,
        datasets={"train": trainloader, "val": testloader, "test": testloader},
        metrics=[acc],
        debug=hparams.debug,        
    )

    m.test(
        datasets={"test": testloader},
    )

    m.export("output_onnx", "onnx", (3, 32, 32))



"""
pretrained = ImageClassification()
pretrained.load_state_dict(torch.load("output/model.pth"))
"""