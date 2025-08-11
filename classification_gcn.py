import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class GCNNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=256, dropout=0.5, freeze_layers=True):
        super(GCNNet, self).__init__()
        # GCN Feature Extraction Layer
        self.conv1 = GCNConv(num_features_xd, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, 32)

        # GCGR-Specific Classification Head
        self.fc_g1 = torch.nn.Linear(32, 1024)
        self.fc_g2 = torch.nn.Linear(1024, 512)
        self.fc_g3 = torch.nn.Linear(512, 256)
        self.out = torch.nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights(freeze_layers)

    def _init_weights(self, freeze_layers):
        #Control Layer Freezing
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            if freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Shared Feature Extraction
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = self.relu(self.conv4(x, edge_index))
        x = gmp(x, batch)

        # GCGR Classification Head
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.relu(self.fc_g2(x))
        x = self.dropout(x)
        last_layer_features = self.relu(self.fc_g3(x))
        x = self.dropout(last_layer_features)
        x = self.out(x)  


        return x, last_layer_features
