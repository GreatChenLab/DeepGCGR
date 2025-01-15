# DeepGCGR
##### DeepGCGR: an Interpretable Two-Layer Deep Learning Model for GCGR Antagonist Discovery
##### In this study, we developed a two-step deep learning-based model, DeepGCGR, for predicting GCGR ligands and classifying them as agonists or antagonists.
![框架11](https://github.com/user-attachments/assets/b8779136-a049-48ce-bfda-986739679dfc)
## Environment
The model relies on PyTorch 1.8.0 and PyTorch Geometric (pyg) 2.5.2.
## Data
##### GCGR ligand activity prediction model dataset（GCGR_P）:GCGR_C_test.csv
##### GCGR ligand agonist/antagonist classification model dataset（GCGR_C）:GCGR_C_train_validation.csv；GCGR_C_test.csv.
##### natural products library（T2DM_TCM）：T2DM_TCM.csv
## Model
##### GCGR ligand activity prediction model（P model）：prediction_gat.py
##### GCGR ligand agonist/antagonist classification model（C model）：classification_gcn.py
## Virtual screening
##### Screening for GCGR high activity ligands using p-modelling：prediction.py
##### Classification of GCGR ligands using the c-model：classification.py
## Other dependency packages
| Package name            | Versions         |
|-------------------------|------------------|
| aiohttp                 | 3.9.5            |
| aiosignal               | 1.3.1            |
| attrs                   | 23.2.0           |
| beautifulsoup4          | 4.12.3           |
| brotli-python           | 1.0.9            |
| charset-normalizer      | 3.3.2            |
| contourpy               | 1.1.1            |
| cycler                  | 0.12.1           |
| et-xmlfile              | 2.0.0            |
| fonttools               | 4.53.0           |
| frozenlist              | 1.4.0            |
| fsspec                  | 2024.6.0         |
| future                  | 1.0.0            |
| idna                    | 3.7              |
| importlib-metadata      | 8.5.0            |
| importlib-resources     | 6.4.0            |
| jinja2                  | 2.11.3           |
| joblib                  | 1.4.2            |
| kiwisolver              | 1.4.5            |
| matplotlib              | 3.7.5            |
| networkx                | 3.1              |
| numba                   | 0.58.1           |
| numpy                   | 1.24.4           |
| openpyxl                | 3.1.5            |
| pandas                  | 2.0.3            |
| pillow                  | 10.3.0           |
| plotly                  | 5.24.1           |
| pyg                     | 2.5.2            |
| pynndescent             | 0.5.13           |
| python-dateutil         | 2.9.0.post0      |
| requests                | 2.32.3           |
| scikit-learn            | 1.3.2            |
| scipy                   | 1.10.1           |
| seaborn                 | 0.13.2           |
| soupsieve               | 2.6              |
| tenacity                | 9.0.0            |
| torch                   | 1.8.0            |
| tqdm                    | 4.66.4           |
| umap-learn              | 0.5.6            |
| urllib3                 | 2.2.1            |
| yarl                    | 1.7.2            |
| zipp                    | 3.19.2           |
