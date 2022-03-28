import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_client_data
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, device,  dataset, data_dir, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        self.K = K
        self.best_acc = 0
        self.early_stopping_counter = 500
        self.personal_learning_rate = personal_learning_rate
        for i in range(num_users):
            train, test = read_client_data(data_dir, i)
            user = UserpFedMe(device, i, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,num_users)
        print("Finished creating pFedMe server.")

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
            # send all parameter for users 
            self.send_parameters()
            
            self.early_stopping_counter -= 1
            if self.early_stopping_counter == 0:  # no improvement is seen for too long, stop early
                break

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()
    
    def evaluate_personalized_model(self):
        accs = []
        losses = []
        for c in self.users:
            acc, num, loss = c.test_persionalized_model()
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
        
        accs = []
        total_correct_count = 0
        total_test_samples = 0
        
        for c in self.users:
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

    def save_model(self):
        for c in self.users:
            c.update_parameters(c.persionalized_model_bar)
            c.save_model()
            c.update_parameters(c.local_model)
    
    def load_model(self):
        for c in self.users:
            c.load_model()
    
  
