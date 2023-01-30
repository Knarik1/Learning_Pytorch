import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import urllib.request
from urllib.error import HTTPError

import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_PATH = os.path.join(os.getcwd(), 'data')
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'saved_models')
SEED = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int=2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # for GPU    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmarks = False


def download_files() -> None:
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial3/"

    # Files to download
    pretrained_files = ["FashionMNIST_elu.config", "FashionMNIST_elu.tar",
                        "FashionMNIST_leakyrelu.config", "FashionMNIST_leakyrelu.tar",
                        "FashionMNIST_relu.config", "FashionMNIST_relu.tar",
                        "FashionMNIST_sigmoid.config", "FashionMNIST_sigmoid.tar",
                        "FashionMNIST_swish.config", "FashionMNIST_swish.tar",
                        "FashionMNIST_tanh.config", "FashionMNIST_tanh.tar"]

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)                    

    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)

        if not os.path.isfile(file_path):
            file_url = os.path.join(base_url, file_name)
            print(f"Downloading {file_name}")

            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(e)   


class ActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {
            "name": self.name
        }


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1+torch.exp(-x))


class Tanh(ActivationFunction):
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class ReLU(ActivationFunction):
    def forward(self, x):
        # faster 0.009
        return torch.max(torch.tensor(0), x)
        # slower 0.013
        # return x * (x>0).float() 
        # slower 0.011
        # return torch.where(x>=0, x, 0)  


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.config['alpha'] = alpha

    def forward(self, x):
        return torch.where(x>=0, x, self.config["alpha"] * x)


class ELU(ActivationFunction):
    def forward(self, x):
        return torch.where(x>=0, x, torch.exp(x) - 1)


class Swish(ActivationFunction):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_grads(act_f, x):
    x = x.clone().requires_grad_()
    out = act_f(x)
    out.sum().backward()

    return x.grad


def get_viz(act_fn, x, ax):
    outs = act_fn(x)
    grads = get_grads(act_fn, x)

    # cpu
    x_cpu, outs_cpu, grads_cpu = x.detach().cpu().numpy(), outs.detach().cpu().numpy(), grads.detach().cpu().numpy()

    # plotting
    ax.plot(x, outs, label="ActFn")
    ax.plot(x, grads, label="GradFn")
    ax.legend()
    ax.set_title(act_fn.name, color="white")


act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish
}       



class BaseNetwork(nn.Module):
    def __init__(self, input_size: int=768, num_classes=10, hidden_sizes: list[int]=[512, 256, 256, 128], act_fn: ActivationFunction = "relu"):
        super().__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes

        for i in range(len(layer_sizes)-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]) , nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-1], num_classes), nn.Softmax(dim=-1)]    
        self.layers = nn.Sequential(*layers)
        print(layers)    


    def forward(self, x):
        x = x.view(x.size[0], -1)
        return self.layers(x)


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")        



if __name__ == "__main__":  
    set_seed(SEED)
    print("Using device:", DEVICE, ", seed:", SEED)

    download_files()

    x = torch.randn((2,3), requires_grad=True)
    t = time.time()

    net = BaseNetwork()
    


    
  











    
