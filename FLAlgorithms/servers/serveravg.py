import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_client_data
import numpy as np

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset, data_dir, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)
        
        self.best_acc = 0
        self.early_stopping_counter = 500

        for i in range(num_users):
            train , test = read_client_data(data_dir, i)
            user = UserAVG(device, i, train, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,num_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.validate()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            self.aggregate_parameters()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
    
    def validate(self):
        accs = []
        losses = []
        for c in self.users:
            acc, num, loss = c.validate()
            accs.append(acc / num)
            losses.append(loss / num)
            
        avg_acc = sum(accs) / len(accs)
        avg_loss = sum(losses) / len(losses)
        
        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
            self.save_model()
            self.early_stopping_counter = 500
        #print("stats_train[1]",stats_train[3][0])
        print("Average Validation Accurancy: {:.4f}".format(avg_acc))
        print("Average Validation Loss: {:.4f}".format(avg_loss))
    
    def test(self):
        self.load_model()
        self.send_parameters()
        
        accs = []
        total_correct_count = 0
        total_test_samples = 0
        
        for c in self.users:
            c.finetune()
            acc, num = c.test()
            accs.append(acc / num)
            total_correct_count += acc
            total_test_samples += num
        
        acc_over_clients = sum(accs)/len(accs)
        acc_over_samples = total_correct_count / total_test_samples
    
        print('------------------------------------------')
        print(f'acc averaging all clients: {acc_over_clients}')
        print(
            f'acc over all test samples: {acc_over_samples}')
        return acc_over_clients, acc_over_samples