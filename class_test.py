import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from class_datacreate import smiles_to_graph_data
from classification_gcn import GCNNet
from class_fine_train import test



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(78)

gcgr_data = pd.read_csv('GCGR_C_fine_train.csv')
gcgr_smiles = gcgr_data['Smiles'].tolist()
gcgr_labels = gcgr_data['antagonist1&agonist0'].tolist()

gcgr_graph_data = smiles_to_graph_data(gcgr_smiles, gcgr_labels)

pretrain_batch_size = 64
fine_tune_batch_size = 32

gcgr_loader = DataLoader(gcgr_graph_data, batch_size=64, shuffle=False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stage 2: Transfer Learning (GCGR Fine-tuning)
    gcgr_train, gcgr_test = train_test_split(gcgr_graph_data, test_size=0.2, random_state=42)

    gcgr_train_loader = DataLoader(gcgr_train, batch_size=fine_tune_batch_size, shuffle=True, drop_last=True)
    gcgr_test_loader = DataLoader(gcgr_test, batch_size=fine_tune_batch_size, shuffle=False, drop_last=True)

    transfer_model = GCNNet(freeze_layers=True).to(device)

    transfer_optimizer = torch.optim.Adam([
        {'params': transfer_model.fc_g1.parameters(), 'lr': 1e-4},
        {'params': transfer_model.fc_g2.parameters(), 'lr': 1e-4},
        {'params': transfer_model.fc_g3.parameters(), 'lr': 1e-4},
        {'params': transfer_model.out.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    #Compute pos_weight for BCEWithLogitsLoss
    y_train_tensor = torch.tensor([data.y.item() for data in gcgr_train])
    num_positive = torch.sum(y_train_tensor).item()
    num_negative = len(y_train_tensor) - num_positive
    pos_weight = torch.tensor(num_negative / (num_positive + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    test_fpr, test_tpr, test_roc_auc, precisions, recalls, aupr = test(
        transfer_model,
        gcgr_test_loader,
        'final_model.pth'
    )
