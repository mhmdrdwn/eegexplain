from tqdm import tqdm
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import torch

from model import CorrChebNetGCN


NUM_CHANNELS = 19
NUM_CLASSES = 2
NUM_NODE_FEATURES = 11
BATCH_SIZE = 200
DEVICE = torch.device("cpu")

def trainer(num_epochs, train_iter):
    print("Training Model....")
    torch.manual_seed(42)
    model = CorrChebNetGCN(in_features=NUM_NODE_FEATURES, num_classes=NUM_CLASSES)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in tqdm(range(1, num_epochs + 1)):
        loss_sum, n = 0.0, 0
        model.train()
        for t, (x, A, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x = x.float()
            A = A.float()
            y_pred = model(x, A)
            loss = loss_func(y_pred, y)
            loss_sum += loss
            loss.backward()
            optimizer.step() 
    return model