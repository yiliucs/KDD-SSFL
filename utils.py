import torch
from torch import nn, autograd
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd
import torchvision
import random
from sklearn import metrics
import copy
from collections import Counter

def data_iid(dataset, num_users):
    num_shards, num_imgs = 40, 200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset[0]))
    labels =dataset[1].int().numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs = idxs_labels[0,:]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def data_noniid(dataset, num_users,args):
    num_shards, num_imgs = 40, 200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset[0]))
    labels =dataset[1].int().numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users
    
    for i in range(num_users):
        rand_set=[]
        label_set=[]
        count=0
        while len(rand_set)<args.local_label:
            rs=random.choice(idx_shard)
            label=int(dataset[1][idxs[rs*num_imgs]])
            count+=1
            if label not in set(label_set) or count>10:
                label_set.append(label)
                rand_set.append(rs)
                idx_shard=list(set(idx_shard) - set(rand_set))

        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def pesudo_label(args, net_g=None, dataset=None, idxs=None, tao=0.99):
    net_g.eval()
    idxs = list(idxs)

    image = dataset[0][idxs]
    label = dataset[1][idxs]
    image = image.numpy()
    label = label.numpy()
    x_pusedo = []
    y_pusedo = []
    y_true=[]
    for i in range(len(image)):
        x_temp = image[i][np.newaxis, :, :]
        x_temp = torch.tensor(torch.from_numpy(x_temp).float())
        x_temp = x_temp.to(args.device)
        with torch.no_grad():
            p_out, output = net_g(x_temp)
            pseudo = torch.softmax(p_out.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo, dim=-1)
            if max_probs > tao:
                x_pusedo.append(x_temp.cpu().numpy())
                y_pusedo.append(int(targets_u.cpu().numpy()))
                y_true.append(label[i])
    x_pusedo = np.array(x_pusedo)
    x_pusedo = np.squeeze(x_pusedo)
    y_pusedo = np.array(y_pusedo)
    y_true = np.array(y_true)
    x_pusedo = torch.from_numpy(x_pusedo).float()
    y_pusedo = torch.from_numpy(y_pusedo)
    y_true = torch.from_numpy(y_true)
    return (x_pusedo, y_pusedo,y_true)

def local_update(args, net, data, target):
    
    traindata =TensorDataset(torch.tensor(data),torch.tensor(target,dtype=torch.long))
    ldr_train = DataLoader(traindata, batch_size = args.local_bs, shuffle=True)
    
    net.train()
    # train and update
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()
            p_out,log_probs = net(images)
            loss =F.nll_loss(log_probs, labels)
            loss.backward()
            optimizer.step()
            if args.verbose and batch_idx % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(images), len(ldr_train.dataset),
                           100. * batch_idx / len(ldr_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = F.nll_loss()
        self.selected_clients = []
        idxs=list(idxs)
        image=dataset[0][idxs]
        label=dataset[1][idxs]
        traindata =TensorDataset(torch.tensor(image),torch.tensor(label,dtype=torch.long))
        self.ldr_train = DataLoader(traindata, batch_size = self.args.local_bs,shuffle=False)
        
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                p_out,log_probs = net(images)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
def global_train(args, net, epoch,dataset=None):
    net.train()
    image=dataset[0]
    label=dataset[1]
    traindata =TensorDataset(torch.tensor(image),torch.tensor(label,dtype=torch.long))
    train_loader= DataLoader(traindata, batch_size = 128,shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    epoch_loss = []
#     for iter in range(epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        net.zero_grad()
        p_out,log_probs = net(images)
        loss = F.nll_loss(log_probs, labels)
        loss.backward()
        optimizer.step()
        if args.verbose and batch_idx % 10 == 0:
            print('Global Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))  

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, (1,3), stride=1,padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2) )) 
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (1,3), stride=1,padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)))
         
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (1,3), stride=1,padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5))
        
        self.fc1 =nn.Sequential(
            nn.Linear(3840, 960),
            nn.Dropout(p=0.5),
            nn.Linear(960,100),
            nn.Dropout(p=0.5),
            nn.Linear(100,5))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 3840)
        x = self.fc1(x)
              
        return x

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test_img(net_g, dataset, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    print("testing dataset length:",len(dataset[0]))
    image=dataset[0]
    label=dataset[1]
    data =TensorDataset(image.clone().detach(),label.clone().detach().long())
    data_loader = DataLoader(data, batch_size=args.bs)
    
#     l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        p_out,log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

# def local_update(args, net, data, target):
    
#     traindata =TensorDataset(torch.tensor(data),torch.tensor(target,dtype=torch.long))
#     ldr_train = DataLoader(traindata, batch_size = args.local_bs, shuffle=True)
    
#     net.train()
#     # train and update
#     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
#     loss_func = nn.CrossEntropyLoss()
#     epoch_loss = []
#     for iter in range(args.local_ep):
#         batch_loss = []
#         for batch_idx, (images, labels) in enumerate(ldr_train):
#             images, labels = images.to(args.device), labels.to(args.device)
#             net.zero_grad()
#             p_out,log_probs = net(images)
#             loss =loss_func(log_probs, labels)
#             loss.backward()
#             optimizer.step()
#             if args.verbose and batch_idx % 10 == 0:
#                 print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     iter, batch_idx * len(images), len(ldr_train.dataset),
#                            100. * batch_idx / len(ldr_train), loss.item()))
#             batch_loss.append(loss.item())
#         epoch_loss.append(sum(batch_loss)/len(batch_loss))
#     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), nn.BatchNorm2d(32))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), nn.BatchNorm2d(64))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 3), stride=1, padding=(0, 1)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), nn.BatchNorm2d(128),
            nn.Dropout(p=0.5))

        self.gru = nn.GRU(input_size=30,
                          hidden_size=16,
                          num_layers=8,
                          batch_first=True)
        self.fla = nn.Flatten()

        self.fc1 = nn.Sequential(nn.Linear(2288, 512), nn.ReLU(),
                                 nn.Dropout(p=0.5), nn.Linear(512, 5))

    def forward(self, x):
        dwt = x[:, 4, :]
        x = x[:, :4, :]
        x = x[:, :, np.newaxis, :]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.squeeze(x, dim=2)
        x, h_n = self.gru(x)
        
        x = self.fla(x)
        x =torch.cat([x,dwt],1)
#         print(x.shape)
        x = self.fc1(x)

        return x,F.log_softmax(x, dim=1)