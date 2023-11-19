import torch
import torch.nn as nn
import torchvision

import os.path
import pickle

import numpy as np
from PIL import Image

from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity

fine_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

fine_labels_index = {}
for i, x in enumerate(fine_labels):
    fine_labels_index[x] = i

class_hierarchy = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
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
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
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

classes_for_parent = {}
for x in new_class_hierarchy.keys():
    classes_for_parent[x] = sorted([j for y in new_class_hierarchy[x] for j in class_hierarchy[y]])

classes_for_parent_index = {}
for i, x in enumerate(classes_for_parent):
    for j, y in enumerate(classes_for_parent[x]):
        classes_for_parent_index[y] = (i*10) +j

mapping = {}
for x in fine_labels_index.keys():
    mapping[fine_labels_index[x]] = classes_for_parent_index[x]

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

#clustering_mapping = {0: 0, 1: 0, 53: 0, 54: 0, 57: 0, 62: 0, 70: 0, 82: 0, 83: 0, 92: 0, 4: 1, 27: 1, 29: 1, 32: 1, 44: 1, 50: 1, 51: 1, 63: 1, 64: 1, 74: 1, 75: 1, 77: 1, 78: 1, 80: 1, 2: 2, 5: 2, 11: 2, 20: 2, 25: 2, 35: 2, 36: 2, 39: 2, 46: 2, 84: 2, 86: 2, 87: 2, 94: 2, 98: 2, 3: 3, 15: 3, 19: 3, 21: 3, 31: 3, 34: 3, 38: 3, 42: 3, 43: 3, 65: 3, 66: 3, 88: 3, 97: 3, 6: 4, 7: 4, 14: 4, 18: 4, 24: 4, 26: 4, 45: 4, 79: 4, 30: 5, 55: 5, 67: 5, 72: 5, 73: 5, 91: 5, 93: 5, 95: 5, 9: 6, 10: 6, 16: 6, 22: 6, 23: 6, 28: 6, 40: 6, 60: 6, 61: 6, 69: 6, 71: 6, 99: 6, 12: 7, 17: 7, 37: 7, 49: 7, 68: 7, 76: 7, 90: 7, 8: 8, 13: 8, 41: 8, 48: 8, 58: 8, 81: 8, 85: 8, 89: 8, 33: 9, 47: 9, 52: 9, 56: 9, 59: 9, 96: 9}
clustering_mapping = {0: 0, 86: 0, 61: 0, 24: 0, 17: 0, 9: 0, 53: 0, 48: 0, 28: 0, 57: 0, 2: 1, 7: 1, 38: 1, 85: 1, 19: 1, 25: 1, 84: 1, 77: 1, 34: 1, 15: 1, 4: 2, 29: 2, 80: 2, 27: 2, 97: 2, 40: 2, 63: 2, 14: 2, 82: 2, 3: 2, 8: 3, 39: 3, 78: 3, 99: 3, 91: 3, 36: 3, 16: 3, 75: 3, 26: 3, 43: 3, 11: 4, 35: 4, 98: 4, 1: 4, 46: 4, 62: 4, 92: 4, 21: 4, 31: 4, 54: 4, 18: 5, 90: 5, 13: 5, 42: 5, 37: 5, 88: 5, 33: 5, 12: 5, 51: 5, 89: 5, 30: 6, 95: 6, 73: 6, 23: 6, 69: 6, 60: 6, 71: 6, 83: 6, 20: 6, 49: 6, 45: 7, 67: 7, 10: 7, 79: 7, 5: 7, 93: 7, 50: 7, 55: 7, 44: 7, 32: 7, 64: 8, 22: 8, 66: 8, 65: 8, 70: 8, 6: 8, 94: 8, 87: 8, 58: 8, 72: 8, 96: 9, 59: 9, 47: 9, 52: 9, 81: 9, 56: 9, 76: 9, 68: 9, 41: 9, 74: 9}


#custom_classes = [x for x in new_class_hierarchy.keys()]
custom_classes = [x for x in clustering_mapping.keys()]

class CIFAR100CUSTOM(torchvision.datasets.CIFAR100):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        coarse: bool = True
    ) -> None:

        self.coarse = coarse

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
                #print(entry["fine_labels"])         

                if(self.coarse):
                    self.targets.extend([parent_10[x] for x in entry["coarse_labels"]])
                else:
                    self.targets.extend([mapping[x] for x in entry["fine_labels"]])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            #print(data['fine_label_names'])
            if(self.coarse):
                self.classes = ["aquatic", "plants", "food", "houshold", "insects", "outdoor_scenes", "large_animals", "medium_animals", "vehicles", "other"]
            else:
                self.classes = classes_for_parent_index.keys()

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}