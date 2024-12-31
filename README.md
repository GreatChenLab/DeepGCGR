# DeepGCGR
DeepGCGR: an Interpretable Two-Layer Deep Learning Model for GCGR Antagonist Discovery
# Environment
The model relies on PyTorch 1.8.0 and PyTorch Geometric (pyg) 2.5.2.
# Data
GCGR ligand activity prediction model dataset（GCGR_P）:GCGR_C_test.csv
GCGR ligand agonist/antagonist classification model dataset（GCGR_C）:GCGR_C_train_validation.csv；GCGR_C_test.csv.
natural products library（T2DM_TCM）：T2DM_TCM.csv
# Model
GCGR ligand activity prediction model（P model）：prediction_gat.py
GCGR ligand agonist/antagonist classification model（C model）：classification_gcn.py
# Virtual screening
Screening for GCGR high activity ligands using p-modelling：prediction.py
Classification of GCGR ligands using the c-model：classification.py
