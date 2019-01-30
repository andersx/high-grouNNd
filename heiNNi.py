#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

##  N   KRR     Large   Smallest 27-5
##  100 5.81    3.6     3.3     
##  200 4.04
##  400 2.86    2.7     2.1
##  800 2.16
## 1106 1.87    2.0     1.6

from __future__ import print_function

import os
import sys
import numpy as np

import qml
import qml.data

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size):

        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size,27)
        # self.fc2 = nn.Linear(input_size,hidden_size)
        self.fc3 = nn.Linear(27,5)
        self.fc4 = nn.Linear(5, 1)
        
    
    def forward(self, x):
        
        x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)

        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = self.fc4(x)

        return x

def test_nn():

    n = int(sys.argv[1])
    
    reps = np.load(sys.argv[2])
    energy = np.load(sys.argv[3]) / 627.51

    keys = list(range(len(energy)))

    print(energy.shape)
    print(energy[0])
    print(reps.shape)
    print(reps[0])

    np.random.shuffle(keys)
    train_keys = keys[:n]
    test_keys = keys[n:]

    # List of representations
    X  = np.array([reps[key] for key in train_keys])
    Xs = np.array([reps[key] for key in test_keys])

    # List of properties
    Y  = np.array([energy[key] for key in train_keys])
    Ys = np.array([energy[key] for key in test_keys])

    dtype = torch.float
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    x = torch.from_numpy(X).to(device, torch.float)
    y = torch.from_numpy(Y.reshape((X.shape[0],1))).to(device, torch.float)

    xs = torch.from_numpy(Xs).to(device, torch.float)
    ys = torch.from_numpy(Ys.reshape((Xs.shape[0],1))).to(device, torch.float)

    model = NeuralNet(X.shape[1]).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100001):

        yt = model(x)

        if (epoch % 1000 == 0): 
            loss =(y - yt).pow(2).sum()
            rmsd = torch.sqrt(loss/X.shape[0]) * 627.51
            mae = (y - yt).abs().sum() * 627.51 / X.shape[0]

            yss = model.forward(xs)
            
            rmsd_s = torch.sqrt((yss - ys).pow(2).sum()/Xs.shape[0]) * 627.51
            mae_s = (yss - ys).abs().sum() * 627.51 / Xs.shape[0]
            print(epoch, mae, rmsd, mae_s, rmsd_s)
       
        loss = criterion(yt, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    test_nn()
