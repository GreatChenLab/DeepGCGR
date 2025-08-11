import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import networkx as nx
from classification_gcn import GCNNet
from class_datacreate import atom_features

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Warning: Failed to parse SMILES: {smile}")
        return None

    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    if len(edges) == 0:
        print(f"Skipping SMILES {smile} due to no edges.")
        return None

    g = nx.Graph(edges).to_directed()
    edge_index = np.array([[e1, e2] for e1, e2 in g.edges], dtype=np.long).T

    if edge_index.shape[0] != 2:
        print(f"Skipping SMILES {smile} due to invalid edge_index shape {edge_index.shape}")
        return None

    return Data(x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))

file_path = "output_predictions.csv"
df = pd.read_csv(file_path)
smiles_list = df['i_SMILES'].tolist()
ids = df['CID'].tolist()

model_path = "final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNNet(freeze_layers=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

graph_data_list = []
valid_smiles_list = []  
valid_ids = []  

for i, smiles in enumerate(smiles_list):
    graph = smile_to_graph(smiles)
    if graph is not None:
        graph.id = ids[i]  
        graph_data_list.append(graph)
        valid_smiles_list.append(smiles) 
        valid_ids.append(ids[i])  


loader = DataLoader(graph_data_list, batch_size=32, shuffle=False)

# classification
predictions = []
ids_result = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)
        outputs, _ = model(data)
        preds = outputs.squeeze().cpu().numpy()
        predictions.extend(preds)
        ids_result.extend(data.id.cpu().numpy())

df['Predicted_Class'] = np.nan 
for id_val, pred in zip(valid_ids, predictions):
    classification = 1 if pred >= 0.5 else 0  
    df.loc[df['CID'] == id_val, 'Predicted_Class'] = classification


df.to_csv("classified_smiles.csv", index=False, encoding='utf-8-sig')
