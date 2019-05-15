import numpy as np 
import pickle 

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt 

import os 

from smooth import smooth 

device = 'cuda:0'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.eeg_1 = nn.Linear(12, 100)
        self.eeg_2 = nn.Linear(100, 50)
        self.eeg = nn.Linear(50, H)
        self.fc2 = nn.Linear(12, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h_eeg = F.relu(self.eeg_1(x))
        h_eeg = F.relu(self.eeg_2(h_eeg))
        h1 = F.relu(self.eeg(h_eeg)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2

class Net_de(nn.Module):
    def __init__(self):
        super(Net_de, self).__init__()
        self.eeg_1 = nn.Linear(310, 100)
        self.eeg_2 = nn.Linear(100, 50)
        self.eeg = nn.Linear(50, H)
        self.fc2 = nn.Linear(33, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h_eeg = F.relu(self.eeg_1(x))
        h_eeg = F.relu(self.eeg_2(h_eeg))
        h1 = F.relu(self.eeg(h_eeg)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2  

de_eeg_dir = './DE/eeg/'
de_eye_dir = './DE/eye/'
trans_data_dir = './trans_data/'

file_names = os.listdir(de_eeg_dir)
file_names.sort() 

# Original DE features
import sklearn.preprocessing as preprocessing 
H=50
n_epoch = 80000
for item in file_names[0]:
    item = file_names[0]
    x_sample = np.load(de_eeg_dir + item )
    y_sample = np.load(de_eye_dir + item )
    x_sample = preprocessing.scale(x_sample)
    y_sample = preprocessing.scale(y_sample)
    model_de = Net_de().to(device)
    optimizer_de = torch.optim.Adam(model_de.parameters(), lr=0.001)
    plot_loss_de = []

    for epoch in tqdm(range(n_epoch)): 
        # selecting batch
        number_perm = torch.randperm(1823)
        idx = number_perm[:500]
        x_sample_used = x_sample[idx]
        y_sample_used = y_sample[idx]
        y_shuffle_used = np.random.permutation(y_sample_used)
        
        # x_sample_used = Variable(torch.from_numpy(x_sample_used).type(torch.FloatTensor), requires_grad = False)
        # y_sample_used = Variable(torch.from_numpy(y_sample_used).type(torch.FloatTensor), requires_grad = False)
        # y_shuffle_used = Variable(torch.from_numpy(y_shuffle_used).type(torch.FloatTensor), requires_grad = False) 
        x_sample_used = torch.from_numpy(x_sample_used).float().to(device)
        y_sample_used = torch.from_numpy(y_sample_used).float().to(device)
        y_shuffle_used = torch.from_numpy(y_shuffle_used).float().to(device)
        
        pred_xy_de = model_de(x_sample_used, y_sample_used)
        pred_x_y_de = model_de(x_sample_used, y_shuffle_used)

        ret_de = torch.mean(pred_xy_de) - torch.log(torch.mean(torch.exp(pred_x_y_de)))
        loss_de = -ret_de  # maximize
        plot_loss_de.append(loss_de.to('cpu').item())
        model_de.zero_grad()
        loss_de.backward()
        optimizer_de.step()

    plot_y_de = np.array(plot_loss_de).reshape(-1,)
    smooth_mi_de = smooth(plot_y_de)

    trans_all = np.load(trans_data_dir + item[:-4]+'.npz')
    x_sample_trans = trans_all['eeg']
    y_sample_trans = trans_all['eye']
    model_trans = Net().to(device)
    optimizer_trans = torch.optim.Adam(model_trans.parameters(), lr=0.001)
    plot_loss_trans = []

    for epoch in tqdm(range(n_epoch)):     
        number_perm = torch.randperm(1823)
        idx = number_perm[:500]
        x_sample_used_trans = x_sample_trans[idx]
        y_sample_used_trans = y_sample_trans[idx]
        y_shuffle_used_trans = np.random.permutation(y_sample_used_trans)
        
        # x_sample_used_trans = Variable(torch.from_numpy(x_sample_used_trans).type(torch.FloatTensor), requires_grad = False)
        # y_sample_used_trans = Variable(torch.from_numpy(y_sample_used_trans).type(torch.FloatTensor), requires_grad = False)
        # y_shuffle_used_trans = Variable(torch.from_numpy(y_shuffle_used_trans).type(torch.FloatTensor), requires_grad = False) 
        x_sample_used_trans = torch.from_numpy(x_sample_used_trans).float().to(device)
        y_sample_used_trans = torch.from_numpy(y_sample_used_trans).float().to(device)
        y_shuffle_used_trans = torch.from_numpy(y_shuffle_used_trans).float().to(device)
        
        pred_xy_trans = model_trans(x_sample_used_trans, y_sample_used_trans)
        pred_x_y_trans = model_trans(x_sample_used_trans, y_shuffle_used_trans)

        ret_trans = torch.mean(pred_xy_trans) - torch.log(torch.mean(torch.exp(pred_x_y_trans)))
        loss_trans = -ret_trans  # maximize
        plot_loss_trans.append(loss_trans.to('cpu').item())
        model_trans.zero_grad()
        loss_trans.backward()
        optimizer_trans.step()

    plot_y_trans = np.array(plot_loss_trans).reshape(-1,)
    smooth_mi_trans = smooth(plot_y_trans)

    fig, ax = plt.subplots()
    # ax.plot(plot_x_2, -plot_y_2, 'b')
    # ax.plot(-smooth_mi_de,'g')
    # ax.plot(-smooth_mi_trans,'r')
    ax.plot(-plot_y_de,'g')
    ax.plot(-plot_y_trans,'r')
    # ax.set_ylim([3,4.5])
    # plt.legend()
    plt.show()
    # plt.savefig('./mi_figs/'+item[:-4]+'.eps')
