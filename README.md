# Overview
This Repository contains the code of the master thesis of Philipp Höfling with the titel "Development of an AI-based algorithm for the classification of gastric tissue in computational pathology". 
Central to this repository is the training pipeline for ResNet18 models and a evaluation pipeline for trained models on Whole Slide Images (WSI).
The dataset and the final models are hosted at the the Otto-Friedrich University Bamberg's chair of Explainable Machine Learning.

# Data
The datasets located at the university chair's file server. Please contact the repository owner or the university chair for access to the server and further instructions on how to access the datasets and the project files.

# Usage
All methods and pipelines are accessed via [main.py](src/main.py), please replace all variables to datsets and anntotaions accordingly to your setup
The hyperparameters and class names for testing are controled via [config.py](src/config.py).

#Technical Overview
All used libaries are contained in environment.yaml.
Pytorch for model training and inference, skikit learn for metrices, Openslide for access to WSI others numpy, pandas, matplot lib, cv2, seaborn, cv2
