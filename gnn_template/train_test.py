import time
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torch.distributed as dist
from config import args
import math
import pandas as pd
import copy
import torch_geometric
import os

def getLossAcc(criterion, output, labels, extra=None):
    if torch.isnan(output).sum():
        print("Error: there is nan in output")
        
    # Loss
    loss = (criterion(output, labels) * extra.weight).sum()

    pred = torch.argmax(output, dim=1)
    acc = (pred==labels)# .sum() / len(pred)
    # # print(output[:10], flush=True)
    # sig = acc[labels==1]
    # print(f"{sig.sum() / len(sig)} len(sig)={len(sig)}", flush=True)
    # sig = acc[labels==0]
    # print(f"{sig.sum() / len(sig)} len(bkg)={len(sig)}", flush=True)
    # print(flush=True)

    return loss, acc, pred


device = args.device
def train_one_epoch(model, trainloader, criterion, optimizer, res):
    model.train()
    
    timeBegin = time.time()
    total_n_data, total_correct, total_loss = torch.tensor(0.).to(device),\
                        torch.tensor(0.).to(device), torch.tensor(0.).to(device)
    total_n_loss = 0
    for i, data in enumerate(trainloader, 0):
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Model output
        data = data.to(device)
        target = data.y
        output = model(data)
        
        # print(output,flush=True)
        loss, acc, _ = getLossAcc(criterion, output, target, extra=data)    
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()

        total_n_loss += len(output)
        total_n_data += len(acc)
        total_correct += acc.sum().item()
        total_loss += loss.item()

    # record result
    if args.distributed:
        dist.barrier()
        for ary in [total_n_data, total_correct, total_loss]:   # collect data from all gpus
            dist.all_reduce(ary, op=dist.ReduceOp.SUM)      
    res['train_time'].append(time.time()-timeBegin)
    res['train_loss'].append(float(total_loss/total_n_loss))
    res['train_acc'].append(float(total_correct/total_n_data))
    

def test_one_epoch(model, testloader, criterion, res, check_output = False, save_new_graph_loc=None, save_new_graph_id=None):
    model.eval()
    timeBegin = time.time()
    # Record scores
    predLog = []
    outputLog = []

    total_n_data, total_correct, total_loss = 0, 0, 0
    total_n_loss = 0
    for i, data in enumerate(testloader, 0):
        data = data.to(device)
        target = data.y

        with torch.no_grad():
            output = model(data)        
            
            loss, acc, pred = getLossAcc(criterion, output, target, extra=data)     

            total_n_loss += len(output)
            total_n_data += len(acc)
            total_correct += acc.sum().item()
            total_loss += loss.item()
            predLog.append(torch.stack((target, pred), 1).detach().cpu().numpy() )
            outputLog.append(torch.cat((target.view(-1,1), output), 1).detach().cpu().numpy() )

    if check_output:
        print(" labels: ", data.y[0:5])
        print(" output: ", output[0:5, :])
        print()

    # record result
    prediction = np.concatenate(predLog, axis=0)
    outputs = np.concatenate(outputLog, axis=0)

    del predLog, outputLog

    res['test_time'].append(time.time()-timeBegin)
    res['test_loss'].append(float(total_loss/total_n_loss))
    res['test_acc'].append(float(total_correct/total_n_data))

    return prediction, outputs
