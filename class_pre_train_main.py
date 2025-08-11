import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from class_datacreate import smiles_to_graph_data
from classification_gcn import GCNNet
from class_pre_train import pretrain





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(78)

train_val_data = pd.read_csv('GCGR_C_pre_train.csv',dtype={'Smiles': str})
train_val_smiles = train_val_data['Smiles'].tolist()
train_val_labels = train_val_data['antagonist1&agonist0'].tolist()

train_val_graph_data = smiles_to_graph_data(train_val_smiles, train_val_labels)



train_graph_data, val_graph_data = train_test_split(train_val_graph_data, test_size=0.2, random_state=42)

pretrain_batch_size = 64
fine_tune_batch_size = 32


train_loader = DataLoader(train_graph_data, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_graph_data, batch_size=64, shuffle=False, drop_last=True)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stage 1: Pre-training
    pretrain_model = GCNNet(freeze_layers=False).to(device)
    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.0001)
    pretrain_criterion = nn.BCEWithLogitsLoss()
    pretrain_path = 'pretrained_model.pth'
    pretrain(pretrain_model, train_loader, val_loader, pretrain_optimizer, pretrain_criterion, epochs=200,
             save_path=pretrain_path)

