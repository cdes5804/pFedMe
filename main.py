#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
from dataset.generate_cifar10 import generate_cifar10
from dataset.generate_mnist import generate_mnist
import torch
import sys
torch.manual_seed(0)

def run(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, class_per_client):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    
    data_dir_path = os.path.join('dataset', dataset)
    if dataset == 'Cifar10':
        data_path, _ = generate_cifar10(data_dir_path, numusers, 10, class_per_client, True, False, None)
    elif dataset == 'Mnist':
        data_path, _ = generate_mnist(data_dir_path, numusers, 10, class_per_client, True, False, None)

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60,10).to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            else:
                model = DNN(60,20,10).to(device), model

        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, data_path, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "pFedMe"):
            server = pFedMe(device, dataset, data_path, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)

        if(algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)

        server.train()
        acc_over_clients, acc_over_samples = server.test()
        
        return acc_over_clients, acc_over_samples
    
def main(cmd_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--class_per_client", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args(cmd_args)

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    return run(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu,
        class_per_client=args.class_per_client
        )


if __name__ == "__main__":
    main(sys.argv[1:])