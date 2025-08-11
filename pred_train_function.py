import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import torch.optim as optim
from prediction_gat import GATNet


def normalize_labels(labels, min_val, max_val):
    labels_array = np.array(labels)
    return (labels_array - min_val) / (max_val - min_val)

def denormalize_labels(labels, min_val, max_val):
    return labels * (max_val - min_val) + min_val

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)



def cross_validate(model, dataset, min_label, max_label, k=5, epochs=200):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_train_losses = []

    for fold_num, (train_index, test_index) in enumerate(kf.split(dataset), 1):
        train_data = [dataset[i] for i in train_index]

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        model = GATNet(num_features_xd=78, output_dim=128, dropout=0.2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        fold_train_loss = []

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            fold_train_loss.append(train_loss)
            print(f'Fold {fold_num}, Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

        fold_train_losses.append(fold_train_loss)
        torch.save(model.state_dict(), f'fold_{fold_num}_model.pth')

