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

from __future__ import print_function

import os
import numpy as np

import qml
import qml.data

from qml.kernels import laplacian_kernel
from qml.math import cho_solve
from qml.representations import get_slatm_mbtypes

import torch
import torch.nn as nn
import torch.nn.functional as F

Q = dict()
Q["H"]  = 0
Q["C"]  = 1
Q["N"]  = 2
Q["O"]  = 3
Q["S"]  = 4


def get_mean_atomic_contribution(mols, feature):

    feature = np.array(feature)

    nm = len(mols)
    nq = len(Q)

    # Make a matrix of number of atoms for each molecule
    at = np.zeros((nm,nq))

    for i, mol in enumerate(mols):
        # print i, mol.atomtypes
        for j, atomtype in enumerate(mol.atomtypes):

            at[i, Q[atomtype]] += 1.0

    # Solve n_atoms x mean_energy_per_atom = energy_for_molecule
    feature_atomic = np.linalg.lstsq(at,feature)[0]

    # Expand into atomic-averaged feature
    feature_mean = np.dot(at, feature_atomic)

    feature_compensated = feature - feature_mean

    feature_compensated -= feature_compensated.mean()

    return feature_compensated, feature_atomic


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, end_size):

        super(NeuralNet, self).__init__()

        # self.fc1 = nn.Linear(input_size, hidden_size) 
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, num_classes)  

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,end_size)
        self.fc4 = nn.Linear(end_size, 1)
    
    def forward(self, x):

        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        
        # return out

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)

        return x


def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])
        hof2 = float(tokens[1])

        energies[xyz_name] = (hof) / 627.51
        #energies[xyz_name] = (hof - hof2) / 627.51

    return energies

def test_nn_bob():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qml.data.Compound() objects
    mols = []


    numbers = dict()

    for xyz_file in sorted(data.keys()):

        # Initialize the qml.data.Compound() objects
        mol = qml.data.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        # mol.generate_eigenvalue_coulomb_matrix()
        mol.generate_coulomb_matrix()
        # mol.generate_bob()
        # print(mol.representation)
        mols.append(mol)

    es = np.array([mol.properties for mol in mols])
    fc, fa = get_mean_atomic_contribution(mols, es)

    for i in range(len(mols)):

        mols[i].properties = fc[i]



    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 2000
    n_train = 4000

    training = mols[:n_train]
    test  = mols[-n_test:]

    # List of representations
    X  = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])


    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    print(X.shape)
    print(Y.shape)

    dtype = torch.float
    device = torch.device("cuda:0")

    N, D_in, H, H2 = n_train, X.shape[1], 50, 10

    x = torch.from_numpy(X).to(device, torch.float)
    y = torch.from_numpy(Y.reshape((N,1))).to(device, torch.float)

    xs = torch.from_numpy(Xs).to(device, torch.float)
    ys = torch.from_numpy(Ys.reshape((n_test,1))).to(device, torch.float)

    model = NeuralNet(X.shape[1], H, H2).to(device)

    # Loss and optimizer
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100000):

        yt = model(x)
        

        if (epoch % 1000 == 0): 
            loss =(y - yt).pow(2).sum()
            rmsd = torch.sqrt(loss/n_train) * 627.51
            mae = (y - yt).abs().sum() * 627.51 / n_train

            yss = model.forward(xs)
            
            rmsd_s = torch.sqrt((yss - ys).pow(2).sum()/n_test) * 627.51
            mae_s = (yss - ys).abs().sum() * 627.51 / n_test
            print(epoch, mae, rmsd, mae_s, rmsd_s)
       
        loss = criterion(yt, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # for epoch in range(100000):

    #     if (epoch % 1000 == 0): 
    #         yt = model(x)
    #     
    #         loss =(y - yt).pow(2).sum()
    #         rmsd = torch.sqrt(loss/n_train) * 627.51
    #         mae = (y - yt).abs().sum() * 627.51 / n_train

    #         yss = model.forward(xs)
    #         
    #         rmsd_s = torch.sqrt((yss - ys).pow(2).sum()/n_test) * 627.51
    #         mae_s = (yss - ys).abs().sum() * 627.51 / n_test
    #         print(epoch, mae, rmsd, mae_s, rmsd_s)
    #     
    #     def closure():

    #         yt = model(x)
    #         return (y - yt).pow(2).sum()

    #     optimizer2.zero_grad()
    #     loss.backward(retain_graph=True)
    #     optimizer2.step(closure)
        
    print(yt[:10]*627.51, y[:10]*627.51)


if __name__ == "__main__":

    test_nn_bob()
