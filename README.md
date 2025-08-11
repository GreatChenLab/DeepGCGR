# DeepGCGR
#### An Interpretable Two-Layer Deep Learning Model for the Discovery of GCGR-Activating Compounds
#### This work presents DeepGCGR, a two-layer deep learning model that integrates graph-based feature representations with Graph Attention Network (GAT) and Graph Convolutional Network (GCN) architectures to predict compounds with potential agonistic activity toward the GCGR. 
<img width="8031" height="5905" alt="figure1" src="https://github.com/user-attachments/assets/4e4f2ced-7b73-458d-9967-b5cf135fb928" />

## Environment
The model relies on PyTorch 1.8.0, python 3.8.18 and PyTorch Geometric (pyg) 2.5.2.
## Data
##### GCGR ligand activity prediction model dataset（GCGR_P）:GCGR_P_train_test.csv
##### GCGR ligand  classification model dataset（GCGR_C）:
##### Pre-training：GCGR_C_pre_train.csv；Fine-tuning：GCGR_C_fine_train.csv
##### natural products library（T2DM_TCM）：T2DM_TCM.csv
## Run model
##### Use the following command to train and test the model.
Pre-training of the GCGR ligand classification model (C model)：python class_pre_train_main.py
Fine-tuning of the GCGR ligand classification model (C model)：python class_pre_train_main.py
Testing of the GCGR ligand classification model (C model)：python class_test.py
Training of the GCGR ligand activity prediction model (P model)：python pred_main.py
Testing of the GCGR ligand activity prediction model (P model)：python pred_test.py
## Virtual screening
##### Use the following command to perform virtual screening with the model.
Predict GCGR ligand activity：python prediction.py
Classify GCGR ligands：python classification.py
## Dependencies
##### Run this command to download the dependencies required for the model：
pip install -r requirements.txt
