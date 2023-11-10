import torch
import torch.nn as nn
import torchvision

import os.path
import pickle

import numpy as np
from PIL import Image

from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity

class_hierarchy = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchids", "poppies", "roses", "sunflowers", "tulips"],
    "food containers": ["bottles", "bowls", "cans", "cups", "plates"],
    "fruit and vegetables": ["apples", "mushrooms", "oranges", "pears", "sweet peppers"],
    "household electrical devices": ["clock", "computer keyboard", "lamp", "telephone", "television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple", "oak", "palm", "pine", "willow"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup truck", "train"],
    "vehicles 2": ["lawn-mower", "rocket", "streetcar", "tank", "tractor"]
}

class_hierarchy_index = {}

i = 0 
for x in class_hierarchy.keys():
    class_hierarchy_index[x] = i
    i+=1


# Mapping the original classes to the new broader categories as specified
new_class_hierarchy = {
    "aquatic": ["aquatic mammals", "fish"],
    "plants": ["flowers", "trees"],
    "food": ["fruit and vegetables", "food containers"],
    "household": ["household electrical devices", "household furniture"],
    "insects": ["insects", "non-insect invertebrates"],
    "outdoor_scenes": ["large natural outdoor scenes", "large man-made outdoor things"],
    "large_animals": ["large carnivores", "large omnivores and herbivores"],
    "medium_animals": ["medium-sized mammals", "small mammals"],
    "vehicles": ["vehicles 1", "vehicles 2"],
    "other": ["people", "reptiles"]
}

i = 0
new_class_hierarchy_index = {}
for x in new_class_hierarchy.keys():
    new_class_hierarchy_index[x] = i
    i+=1

parent_10 = {}

i = 0
for x in new_class_hierarchy.keys():
    for y in new_class_hierarchy[x]:
        parent_10[class_hierarchy_index[y]] = i
    i+=1

#print(new_class_hierarchy)

custom_classes = [x for x in new_class_hierarchy.keys()]

class CIFAR100CUSTOM(torchvision.datasets.CIFAR100):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])                

                coarse_10 = [parent_10[x] for x in entry["coarse_labels"]]

                self.targets.extend(coarse_10)                

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")

            self.classes = ["aquatic", "plants", "food", "houshold", "insects", "outdoor_scenes", "large_animals", "medium_animals", "vehicles", "other"]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}