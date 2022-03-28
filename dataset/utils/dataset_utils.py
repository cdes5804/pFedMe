import os
import ujson
import numpy as np
from sklearn.model_selection import train_test_split

batch_size = 4
train_size = 0.75
least_samples = batch_size / (1-train_size)
sigma = 0.1
beta = 0.5

def check(config_path, train_path, test_path, num_clients, num_labels, niid=False, 
        real=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_labels'] == num_labels and \
            config['non_iid'] == niid and \
            config['real_world'] == real and \
            config['partition'] == partition:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_labels, niid=False, real=True, partition=None, 
                class_per_client=2):
    if num_clients * class_per_client % num_labels != 0:
        print(f'Cannot distribute data with {num_clients} clients and {class_per_client} class per client')
        exit(1)
        
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    
    dataset = []
    for i in range(num_labels):
        idx = dataset_label == i
        dataset.append(dataset_content[idx])
    
    start_user = 0
    num_client_per_class = num_clients * class_per_client // num_labels
    class_distribution_for_clients = np.zeros((num_clients, num_labels))
    for i in range(num_labels):
        chosen_users = [idx % num_clients for idx in range(start_user, start_user + num_client_per_class)]
        num_samples_per_user = len(dataset[i]) // len(chosen_users)
        sample_remainder = len(dataset[i]) % len(chosen_users)
        start_sample = 0
        for user in chosen_users:
            class_distribution_for_clients[user, i] += 1 / len(chosen_users)
            num_samples_for_user = num_samples_per_user
            if user < sample_remainder:
                num_samples_for_user += 1
            X[user] += dataset[i][start_sample: start_sample+num_samples_for_user].tolist()
            y[user] += (i*np.ones(num_samples_for_user)).tolist()
            start_sample += num_samples_for_user
            statistic[user].append((i, num_samples_for_user))
        start_user = (start_user + num_client_per_class) % num_clients

    del data

    # calculate EMD
    global_distribution = np.ones(num_labels) / num_labels
    emd = sum(np.abs(class_distribution_for_clients[0] - global_distribution)) / 2
    
    print(f'EMD: {emd}')

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic, emd


def split_data(X, y, train_size=train_size):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        if min(count) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True, stratify=y[i])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True, stratify=None)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_labels, statistic, niid=False, real=True, partition=None, emd=-1):
    config = {
        'num_clients': num_clients, 
        'num_labels': num_labels, 
        'non_iid': niid, 
        'real_world': real, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic,
        'emd': emd, 
    }

    for idx, train_dict in enumerate(train_data):
        with open(train_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(train_dict, f)
    for idx, test_dict in enumerate(test_data):
        with open(test_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(test_dict, f)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")

def read_emd_from_config(config_path):
    with open(config_path, 'r') as f:
        j = ujson.load(f)
        return j['emd']