import torch
import torch.nn as nn
import torchvision

import os.path
import pickle

import numpy as np
from PIL import Image

from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity

# fine_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# fine_labels_index = {}
# for i, x in enumerate(fine_labels):
#     fine_labels_index[x] = i

# class_hierarchy = {
#     "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
#     "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
#     "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
#     "food containers": ["bottle", "bowl", "can", "cup", "plate"],
#     "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
#     "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
#     "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
#     "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
#     "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
#     "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
#     "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
#     "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
#     "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
#     "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
#     "people": ["baby", "boy", "girl", "man", "woman"],
#     "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
#     "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
#     "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
#     "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
#     "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
# }

# class_hierarchy_index = {}

# i = 0 
# for x in class_hierarchy.keys():
#     class_hierarchy_index[x] = i
#     i+=1


# # Mapping the original classes to the new broader categories as specified
# new_class_hierarchy = {
#     "aquatic": ["aquatic mammals", "fish"],
#     "plants": ["flowers", "trees"],
#     "food": ["fruit and vegetables", "food containers"],
#     "household": ["household electrical devices", "household furniture"],
#     "insects": ["insects", "non-insect invertebrates"],
#     "outdoor_scenes": ["large natural outdoor scenes", "large man-made outdoor things"],
#     "large_animals": ["large carnivores", "large omnivores and herbivores"],
#     "medium_animals": ["medium-sized mammals", "small mammals"],
#     "vehicles": ["vehicles 1", "vehicles 2"],
#     "other": ["people", "reptiles"]
# }

# classes_for_parent = {}
# for x in new_class_hierarchy.keys():
#     classes_for_parent[x] = sorted([j for y in new_class_hierarchy[x] for j in class_hierarchy[y]])

# classes_for_parent_index = {}
# for i, x in enumerate(classes_for_parent):
#     for j, y in enumerate(classes_for_parent[x]):
#         classes_for_parent_index[y] = (i*10) +j

# mapping = {}
# for x in fine_labels_index.keys():
#     mapping[fine_labels_index[x]] = classes_for_parent_index[x]

# i = 0
# new_class_hierarchy_index = {}
# for x in new_class_hierarchy.keys():
#     new_class_hierarchy_index[x] = i
#     i+=1

# parent_10 = {}

# i = 0
# for x in new_class_hierarchy.keys():
#     for y in new_class_hierarchy[x]:
#         parent_10[class_hierarchy_index[y]] = i
#     i+=1



#clustering_mapping = {0: 0, 57: 0, 83: 0, 53: 0, 10: 0, 92: 0, 61: 0, 40: 0, 55: 0, 22: 0, 2: 1, 11: 1, 35: 1, 98: 1, 46: 1, 32: 1, 65: 1, 36: 1, 25: 1, 50: 1, 4: 2, 72: 2, 74: 2, 3: 2, 63: 2, 27: 2, 80: 2, 64: 2, 77: 2, 93: 2, 7: 3, 24: 3, 79: 3, 6: 3, 14: 3, 18: 3, 44: 3, 26: 3, 45: 3, 82: 3, 12: 4, 37: 4, 90: 4, 76: 4, 17: 4, 68: 4, 85: 4, 69: 4, 81: 4, 71: 4, 15: 5, 19: 5, 31: 5, 29: 5, 43: 5, 38: 5, 34: 5, 97: 5, 84: 5, 42: 5, 20: 6, 5: 6, 94: 6, 91: 6, 87: 6, 41: 6, 95: 6, 28: 6, 75: 6, 99: 6, 30: 7, 73: 7, 67: 7, 1: 7, 23: 7, 49: 7, 56: 7, 39: 7, 66: 7, 21: 7, 52: 8, 47: 8, 59: 8, 96: 8, 33: 8, 60: 8, 70: 8, 51: 8, 13: 8, 58: 8, 89: 9, 48: 9, 8: 9, 88: 9, 78: 9, 62: 9, 9: 9, 54: 9, 16: 9, 86: 9}
#clustering_mapping = {0: 0, 57: 0, 83: 0, 53: 0, 10: 0, 92: 0, 40: 0, 77: 0, 22: 0, 61: 0, 2: 1, 11: 1, 35: 1, 98: 1, 46: 1, 32: 1, 36: 1, 65: 1, 25: 1, 50: 1, 4: 2, 55: 2, 72: 2, 74: 2, 3: 2, 63: 2, 27: 2, 64: 2, 80: 2, 21: 2, 7: 3, 24: 3, 79: 3, 6: 3, 14: 3, 18: 3, 44: 3, 26: 3, 45: 3, 93: 3, 12: 4, 37: 4, 90: 4, 76: 4, 17: 4, 85: 4, 68: 4, 69: 4, 81: 4, 5: 4, 16: 5, 9: 5, 28: 5, 86: 5, 94: 5, 87: 5, 88: 5, 66: 5, 84: 5, 41: 5, 23: 6, 71: 6, 49: 6, 60: 6, 73: 6, 67: 6, 30: 6, 43: 6, 33: 6, 39: 6, 31: 7, 19: 7, 15: 7, 29: 7, 38: 7, 97: 7, 75: 7, 59: 7, 96: 7, 13: 7, 52: 8, 47: 8, 56: 8, 82: 8, 42: 8, 70: 8, 91: 8, 51: 8, 58: 8, 89: 8, 99: 9, 78: 9, 8: 9, 95: 9, 34: 9, 1: 9, 48: 9, 20: 9, 54: 9, 62: 9}
#clustering_mapping = {79: 0, 47: 0, 8: 0, 46: 0, 24: 0, 70: 0, 52: 0, 54: 0, 20: 0, 41: 0, 69: 1, 11: 1, 59: 1, 68: 1, 7: 1, 60: 1, 35: 1, 33: 1, 96: 1, 17: 1, 82: 2, 37: 2, 99: 2, 18: 2, 40: 2, 19: 2, 6: 2, 89: 2, 53: 2, 12: 2, 2: 3, 98: 3, 92: 3, 14: 3, 5: 3, 57: 3, 81: 3, 9: 3, 86: 3, 25: 3, 15: 4, 29: 4, 23: 4, 49: 4, 90: 4, 62: 4, 44: 4, 56: 4, 71: 4, 84: 4, 16: 5, 22: 5, 0: 5, 45: 5, 39: 5, 28: 5, 77: 5, 91: 5, 43: 5, 87: 5, 76: 6, 31: 6, 58: 6, 38: 6, 34: 6, 42: 6, 65: 6, 36: 6, 13: 6, 85: 6, 26: 7, 48: 7, 61: 7, 83: 7, 78: 7, 94: 7, 88: 7, 75: 7, 64: 7, 1: 7, 30: 8, 97: 8, 73: 8, 32: 8, 50: 8, 66: 8, 95: 8, 93: 8, 27: 8, 3: 8, 51: 9, 10: 9, 21: 9, 74: 9, 55: 9, 72: 9, 67: 9, 4: 9, 63: 9, 80: 9}
#clustering_mapping = {69: 0, 12: 0, 54: 0, 76: 0, 97: 0, 64: 0, 9: 0, 85: 0, 73: 0, 47: 0, 74: 1, 45: 1, 6: 1, 4: 1, 26: 1, 36: 1, 17: 1, 81: 1, 82: 1, 33: 1, 48: 2, 98: 2, 77: 2, 16: 2, 67: 2, 80: 2, 95: 2, 0: 2, 99: 2, 83: 2, 51: 3, 75: 3, 44: 3, 79: 3, 13: 3, 41: 3, 28: 3, 24: 3, 63: 3, 34: 3, 35: 4, 56: 4, 37: 4, 84: 4, 2: 4, 15: 4, 87: 4, 10: 4, 88: 4, 89: 4, 61: 5, 42: 5, 43: 5, 30: 5, 70: 5, 62: 5, 91: 5, 68: 5, 19: 5, 55: 5, 53: 6, 27: 6, 7: 6, 60: 6, 22: 6, 86: 6, 20: 6, 8: 6, 38: 6, 49: 6, 92: 7, 93: 7, 94: 7, 72: 7, 40: 7, 58: 7, 1: 7, 96: 7, 21: 7, 29: 7, 52: 8, 18: 8, 5: 8, 11: 8, 59: 8, 71: 8, 14: 8, 46: 8, 65: 8, 57: 8, 90: 9, 25: 9, 78: 9, 50: 9, 66: 9, 23: 9, 31: 9, 32: 9, 3: 9, 39: 9}
clustering_mapping = {0: 5, 1: 5, 2: 3, 3: 7, 4: 7, 5: 4, 6: 6, 7: 6, 8: 8, 9: 3, 10: 4, 11: 3, 12: 0, 13: 8, 14: 6, 15: 7, 16: 4, 17: 0, 18: 6, 19: 7, 20: 3, 21: 7, 22: 4, 23: 0, 24: 6, 25: 4, 26: 5, 27: 1, 28: 4, 29: 2, 30: 1, 31: 7, 32: 3, 33: 2, 34: 9, 35: 3, 36: 3, 37: 8, 38: 9, 39: 0, 40: 3, 41: 8, 42: 9, 43: 9, 44: 5, 45: 5, 46: 3, 47: 2, 48: 8, 49: 0, 50: 9, 51: 2, 52: 2, 53: 5, 54: 6, 55: 7, 56: 2, 57: 5, 58: 8, 59: 2, 60: 0, 61: 4, 62: 6, 63: 9, 64: 7, 65: 9, 66: 7, 67: 1, 68: 0, 69: 0, 70: 6, 71: 0, 72: 1, 73: 1, 74: 1, 75: 7, 76: 0, 77: 1, 78: 5, 79: 6, 80: 9, 81: 8, 82: 2, 83: 5, 84: 4, 85: 8, 86: 4, 87: 4, 88: 9, 89: 8, 90: 8, 91: 1, 92: 6, 93: 1, 94: 2, 95: 1, 96: 2, 97: 9, 98: 3, 99: 5}


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
                    self.targets.extend([clustering_mapping[x] for x in entry["fine_labels"]])
                    #self.targets.extend(entry["coarse_labels"])
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
                #self.classes = data['coarse_label_names']
                self.classes = ["aquatic", "plants", "food", "houshold", "insects", "outdoor_scenes", "large_animals", "medium_animals", "vehicles", "other"]
            else:
                self.classes = classes_for_parent_index.keys()

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}