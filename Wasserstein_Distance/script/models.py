from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, overload, Literal
from math import ceil
import torch
from torch import Tensor
from utils import return_act

def initialize_mlp(input_sz, hidden_sz, output_sz, layers,slope, batch_norm, activation):
    func = return_act(act_name=activation,slope=slope)
    phi_layers = []
    phi_layers.append(nn.Linear(input_sz, hidden_sz))
    phi_layers.append(func())
    if batch_norm:
        phi_layers.append(nn.BatchNorm1d(input_sz))
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Linear(hidden_sz, hidden_sz))
            phi_layers.append(func())
            if batch_norm:
                phi_layers.append(nn.BatchNorm1d(hidden_sz))
        else:
            phi_layers.append(nn.Linear(hidden_sz, output_sz))
            phi_layers.append(func())
    phi = nn.Sequential(*phi_layers)
    for layer in phi:
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight) 
    return phi

class SharedMLP(nn.Module):
    def __init__(self,input_sz, hidden_sz, output_sz, layers,slope,batch_norm, activation):
        super(SharedMLP, self).__init__()
        func = return_act(act_name=activation,slope=slope)
        phi_layers = nn.ModuleList()
        current_layer = nn.Sequential()
        self.common_layer = nn.ParameterList()
        current_layer.append(nn.Linear(input_sz, hidden_sz))
        current_layer.append(func())
        if batch_norm:
            current_layer.append(nn.BatchNorm1d(hidden_sz))
        param = nn.Parameter(torch.empty(input_sz, hidden_sz))
        self.common_layer.append(param)
        phi_layers.append(current_layer)
        for i in range(layers - 1):
            current_layer = nn.Sequential()
            if i < layers - 2:
                param = nn.Parameter(torch.empty(hidden_sz, hidden_sz))
                self.common_layer.append(param)
                current_layer.append(nn.Linear(hidden_sz, hidden_sz))
                current_layer.append(func())
                if batch_norm:
                   current_layer.append(nn.BatchNorm1d(hidden_sz))
                phi_layers.append(current_layer)
            else:
                current_layer.append(nn.Linear(hidden_sz, output_sz))
                current_layer.append(func())
                phi_layers.append(current_layer)
                param = nn.Parameter(torch.empty(hidden_sz, output_sz))
                self.common_layer.append(param)
        self.layer = phi_layers
        self.init_params()
    
    def init_params(self):
        for param in self.common_layer:
            torch.nn.init.xavier_uniform_(param)

        for layers in self.layer:
            for layer in layers:
                if type(layer) == nn.Linear:
                    torch.nn.init.xavier_uniform_(layer.weight)     

    def forward(self,input_1:Tensor,input_2:Tensor):
        for layer_id, layer in enumerate(self.layer):
            shared = (input_1.mean(0) + input_2.mean(0))@self.common_layer[layer_id]
            input_1 = layer(input_1) + shared.view(1,-1)
            input_2 = layer(input_2) + shared.view(1,-1)
        input_1, input_2 = input_1.mean(0), input_2.mean(0)
        return input_1, input_2

class SharedProductNet(nn.Module):
    def __init__(self, encoder_params: dict, phi_params: dict, rho_params: dict,activation:str,bn:bool,slope:float):
        super(SharedProductNet, self).__init__()
        self.encoder = SharedMLP(encoder_params['input_dim'], encoder_params['hidden'], encoder_params['output'], encoder_params['layers'], activation=activation,batch_norm=bn,slope=slope)
        self.phi =  initialize_mlp(encoder_params['output'], phi_params['hidden'], phi_params['output'], phi_params['layers'], activation=activation,batch_norm=False,slope=slope)
        self.rho = initialize_mlp(phi_params['output'], rho_params['hidden'], 1, rho_params['layers'], activation=activation,batch_norm=False,slope=slope)

    def forward(self, input1, input2):
        embd1, embd2 = self.encoder(input1,input2)
        embd1, embd2 = self.phi(embd1), self.phi(embd2)
        out = embd1 + embd2
        out = self.rho(out)
        return out, embd1, embd2

