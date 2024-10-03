import numpy as np
import torch
import copy
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import SharedProductNet
from tqdm import trange
import argparse

MSELOSS = nn.MSELoss(reduction='mean')

def train_point_productnet(train_dataset: Dataset,val_dataset, dimension: int, initial: dict,
                           phi: dict, rho: dict, device: str, lr, name: str,factor:float,slope,
                           activation='relu',  iterations=200,
                           batch_size=64, batch = True,wd = 0.0,opt_type ='Adam'):
    embedding_size = phi['output']
    best_loss = np.inf
    best_model = None
    initial['input_dim'] = dimension                            
    model = SharedProductNet(initial, phi, rho,activation=activation,bn=batch,slope=slope)
    model.to(device)
    if opt_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr,weight_decay=wd)
    else:
        optimizer = AdamW(model.parameters(), lr=lr,weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,factor=factor,patience=1)
    epoch_losses = []
    for epoch in trange(iterations):
        optimizer.zero_grad()
        epoch_loss = 0
        for i in trange(len(train_dataset)):
            input1 = train_dataset[i][0].type(torch.float32).to(device)
            input2 = train_dataset[i][1].type(torch.float32).to(device)
            yval = train_dataset[i][2].type(torch.float32).to(device)
            pred, feat1, feat2 = model(input1, input2)
            yval = torch.unsqueeze(yval, dim=0)
            loss = 1 / batch_size * MSELOSS(pred, yval)
            epoch_loss += loss.detach()
            loss.backward()
            if (i != 0 and i % batch_size == 0) or i == len(train_dataset) - 1:
                optimizer.step()
                optimizer.zero_grad()
        
        val_loss,_ = validation_loss(val_dataset=val_dataset,model=model,device='cuda')
        scheduler.step(epoch_loss)
        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = val_loss
            best_epoch = epoch
        print(val_loss)

        epoch_losses.append(epoch_loss / len(train_dataset))

    return best_model, best_epoch


def validation_loss(val_dataset: Dataset, model: SharedProductNet, device: str, image=False):
    total_loss = []
    # model.eval()
    for i in range(len(val_dataset)):
        input1 = val_dataset[i][0].to(device)
        input2 = val_dataset[i][1].to(device)
        if image:
            input1 = torch.unsqueeze(input1, dim=0)
            input2 = torch.unsqueeze(input2, dim=0)
        yval = torch.tensor(val_dataset[i][2])
        pred, _, _ = model(input1, input2)
        if yval > 0.0:
            loss = torch.sum(torch.abs(pred - yval) / yval)
            total_loss.append(loss.detach().item())
    return np.mean(total_loss), np.std(total_loss)
