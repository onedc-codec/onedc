import torch
import os, json, importlib
import random as rd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.transforms import functional as F


class ResizeIfSmall(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        if height < self.patch_size or width < self.patch_size:
            img = F.resize(img, self.patch_size)
        return img


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_datasets(config: dict):
    """
    yaml config example:
    dataset_1:
        target: torchvision.datasets.CIFAR10
        params:
            root: data
            train: True
            download: True
    dataset_2:
        target: torchvision.datasets.CIFAR10
        params:
            root: data
            train: False
            download: True
    """
    datasets = []
    for k, v in config.items():
        item = instantiate_from_config(v)
        datasets.append(item)
    return ConcatDataset(datasets)