import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file, read_emd_from_config

def generate_mnist(dir_path, num_clients, num_labels, class_per_client, niid=False, real=True, partition=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    random.seed(64)
    np.random.seed(64)

    dir_path_for_task = os.path.join(
        dir_path, f'{num_clients}_{class_per_client}')

    # Setup directory for train/test data
    config_path = os.path.join(dir_path_for_task, 'config.json')
    train_path = os.path.join(dir_path_for_task, 'train', 'train.json')
    test_path = os.path.join(dir_path_for_task, 'test', 'test.json')

    if check(config_path, train_path, test_path, num_clients, num_labels, niid, real, partition):
        emd = read_emd_from_config(config_path)
        return dir_path_for_task, emd

    # Get MNIST data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=os.path.join(dir_path, "rawdata"), train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=os.path.join(dir_path, "rawdata"), train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic, emd = separate_data((dataset_image, dataset_label), num_clients, num_labels,
                                    niid, real, partition, class_per_client=class_per_client)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_labels,
              statistic, niid, real, partition, emd)

    return dir_path_for_task, emd