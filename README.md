# Perturbed-GE-LDM

Title: **Predicting Drug-Induced Transcriptional Responses Using Latent Diffusion Model**

Author: ***Chaewon Kim, Sunyong Yoo***

## Datasets
### LINCS L1000 for drug-induced transcriptional profiles
The L1000 was downloaded from the Gene Expression Omnibus with the accession number ([GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)).  
You can also access the preprocessed LINCS L1000 data through the PRnet repository:  
https://github.com/Perturbation-Response-Prediction/PRnet

### IC50 data for drug sensitivity
The IC50 data was downloaded from the GDSC2 website (https://www.cancerrxgene.org/).

## Requirements
Please check the `environment.yaml` file for the full list of dependencies. The main dependencies are:  
- python 3.12.7
- pytorch 2.4.1
- pytorch-cuda 12.4
- transformers 4.51.3
```
conda env create -f environment.yaml
conda activate pert_ldm
```


## Train model
- `train_model.py` trains the Latent Diffusion Model for drug-induced transcriptional response prediction.
- This code is based on PyTorch Data Distributed Processing (DDP) and supports multi-GPU usage.
```
python train_model.py
```

## Note
All of the code will be updated soon.

## Contact
For any questions, please open a GitHub issue or email to chaewonk215@gmail.com.