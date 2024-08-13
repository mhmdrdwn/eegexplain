import torch

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import copy
import networkx as nx
import numpy as np

def uniform_all_data(train_graphs, test_graphs):
    out_train = copy.deepcopy(train_graphs)
    out_test = copy.deepcopy(test_graphs)
    print(train_graphs.shape, test_graphs.shape)
    for band_idx in range(train_graphs.shape[1]):
        for sym_idx in range(train_graphs.shape[2]):
            min_A = train_graphs[:, band_idx, sym_idx].min()
            max_A = train_graphs[:, band_idx, sym_idx].max()
            out_train[:, band_idx, sym_idx] = (train_graphs[:, band_idx, sym_idx] - min_A)/(max_A - min_A)
            out_test[:, band_idx, sym_idx] = (test_graphs[:, band_idx, sym_idx] - min_A)/(max_A - min_A)

    return out_train, out_test


def uniform_separate_A(train_graphs):
    out_normalized = copy.deepcopy(train_graphs)

    for i, A in enumerate(train_graphs):
        
        for j, feat in enumerate(A):
            min_A = train_graphs[i, j].min()
            max_A = train_graphs[i, j].max()
     
            out_normalized[i, j] = (train_graphs[i, j] - min_A)/(max_A - min_A)

    return out_normalized
 

def uniform(graphs):
    out = copy.deepcopy(graphs)

    min_A = graphs.min()
    max_A = graphs.max()
    out = (graphs - min_A)/(max_A - min_A)

    return out

def data_loader(features, graphs, labels, device, batch_size, shuffle=True):
    features, graphs, labels = torch.Tensor(features), torch.Tensor(graphs), torch.Tensor(labels)
    data = torch.utils.data.TensorDataset(features, graphs, labels)
    del features
    del labels
    data_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=shuffle)
    del data
    return data_iter

def standardize_data(train_X, test_X):
 
    train_X_std = copy.deepcopy(train_X)
    test_X_std = copy.deepcopy(test_X)
    
    for i in tqdm(range(train_X.shape[1])):
        for j in range(train_X.shape[2]):
            min_ = np.min(train_X[:, i, j])
            max_ = np.max(train_X[:, i, j])
            train_X_std[:, i, j] = (train_X[:, i, j] - min_)/(max_ - min_)
            test_X_std[:, i, j] = (test_X[:, i, j] - min_)/(max_ - min_)

    return train_X_std, test_X_std

import copy

def uniform_all_data(train_graphs, test_graphs):
    out_train = copy.deepcopy(train_graphs)
    out_test = copy.deepcopy(test_graphs)
    print(train_graphs.shape, test_graphs.shape)
    for band_idx in range(train_graphs.shape[1]):
        #for sym_idx in range(train_graphs.shape[2]):
        min_A = train_graphs[:, band_idx, :].min()
        max_A = train_graphs[:, band_idx, :].max()
        out_train[:, band_idx, :] = (train_graphs[:, band_idx, :] - min_A)/(max_A - min_A)
        out_test[:, band_idx, :] = (test_graphs[:, band_idx, :] - min_A)/(max_A - min_A)

    return out_train, out_test


def uniform_separate_A(train_graphs):
    out_normalized = copy.deepcopy(train_graphs)

    for i, A in enumerate(train_graphs):
        
        for j, feat in enumerate(A):
            min_A = train_graphs[i, j].min()
            max_A = train_graphs[i, j].max()
     
            out_normalized[i, j] = (train_graphs[i, j] - min_A)/(max_A - min_A)

    return out_normalized
 

def uniform(graphs):
    out = copy.deepcopy(graphs)

    min_A = graphs.min()
    max_A = graphs.max()
    out = (graphs - min_A)/(max_A - min_A)

    return out

def normalize_all_A(all_bands_adj):
    all_A = []
    for adj_band in all_bands_adj:
        all_A_band = [] 
        for adj in adj_band:
            all_A_band.append(normalize_A(adj))
        all_A.append(all_A_band)
    all_A = np.array(all_A)
    #print(all_A)
    return all_A

def normalize_A(A):
    norm = []
    A1 = (A + A.T)/2
    A2 = (A - A.T)/2
    A1 = uniform(A1)
    A2 = uniform(A2)
    for graph_feat in [A1, A2]:
        graph_feat = nx.Graph(graph_feat)
        norm.append(nx.normalized_laplacian_matrix(graph_feat).toarray()) 
    return norm


import copy
from tqdm import tqdm

import copy

def uniform(train_graphs, test_graphs):
    out_train = copy.deepcopy(train_graphs)
    out_test = copy.deepcopy(test_graphs)

    min_A = train_graphs.min()
    max_A = train_graphs.max()
    out_train = (train_graphs - min_A)/(max_A - min_A)
    out_test = (test_graphs - min_A)/(max_A - min_A)

    return out_train, out_test

def uniform_corr(train_graphs, test_graphs):
    out_train = copy.deepcopy(train_graphs)
    out_test = copy.deepcopy(test_graphs)
    
    for band_idx in range(5):
        min_A = train_graphs[:, band_idx, :, :].min()
        max_A = train_graphs[:, band_idx, :, :].max()
        out_train[:, band_idx, :, :] = (train_graphs[:, band_idx, :, :] - min_A)/(max_A - min_A)
        out_test[:, band_idx, :, :] = (test_graphs[:, band_idx, :, :] - min_A)/(max_A - min_A)

    return out_train, out_test


def standardize_data(train_X, test_X):
 
    train_X_std = copy.deepcopy(train_X)
    test_X_std = copy.deepcopy(test_X)
    
    for i in tqdm(range(train_X.shape[1])):
        for j in range(train_X.shape[2]):
            min_ = np.min(train_X[:, i, j])
            max_ = np.max(train_X[:, i, j])
            train_X_std[:, i, j] = (train_X[:, i, j] - min_)/(max_ - min_)
            test_X_std[:, i, j] = (test_X[:, i, j] - min_)/(max_ - min_)
    return train_X_std, test_X_std


def label2skip(train_graphs, train_X, train_y, skip_label):
    train_graphs_, train_X_, train_y_ =  [], [], []
    for g, x, y in zip(train_graphs, train_X, train_y):
        if y[0] == skip_label:
            pass
        else:
            train_graphs_.append(g)
            train_X_.append(x)
            train_y_.append(y)
       
    if skip_label == 0:
        train_y_ = [y-1 for y in train_y_]
    elif skip_label == 1:
        train_y_ = [y-1 if y[0]==2 else y for y in train_y_]

    train_graphs, train_X, train_y = train_graphs_, train_X_, train_y_
    train_graphs, train_X, train_y = np.array(train_graphs), np.array(train_X), np.array(train_y)
    return train_graphs, train_X, train_y

def label2skip(train_graphs, train_X, train_y, skip_label):
    train_graphs_, train_X_, train_y_ =  [], [], []
    for g, x, y in zip(train_graphs, train_X, train_y):
        if y[0] == skip_label:
            pass
        else:
            train_graphs_.append(g)
            train_X_.append(x)
            train_y_.append(y)
       
    if skip_label == 0:
        train_y_ = [y-1 for y in train_y_]
    elif skip_label == 1:
        train_y_ = [y-1 if y[0]==2 else y for y in train_y_]

    train_graphs, train_X, train_y = train_graphs_, train_X_, train_y_
    train_graphs, train_X, train_y = np.array(train_graphs), np.array(train_X), np.array(train_y)
    return train_graphs, train_X, train_y


import networkx as nx

def get_lap(graphs):
    graphs_normalized = []
    for graph in tqdm(graphs):
        bands_graphs = []
        for band_graph in graph:
            graph = nx.Graph(band_graph)
            bands_graphs.append(nx.normalized_laplacian_matrix(graph).toarray())
        graphs_normalized.append(bands_graphs)
    return np.array(graphs_normalized)