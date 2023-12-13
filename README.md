# Overview
This Repository contains the code of the master thesis of Philipp Höfling with the titel "Development of an AI-based algorithm for the classification of gastric tissue in computational pathology". 
Central to this repository is a specifically tailored for gastric mucuso training pipeline for ResNet18 models.
As viusalized below the WSI test pipeline allows trained ResNet18s to classify WSIs and output a prelimnary diagnosis and a heatmap of the tissue.
![PipelineResult](https://github.com/PhilippHoefling/WSI-Gastric-Tissue-Classification/assets/40239939/19b6c188-6f6f-4319-a1a3-65a57409be17)
The dataset and the final models are hosted at the the Otto-Friedrich University Bamberg's chair of Explainable Machine Learning.
# Data
The datasets located at the university chair's file server. Please contact the repository owner or the university chair for access to the server and further instructions on how to access the datasets and the project files.

# Usage
All methods and pipelines are accessed via [main.py](src/main.py). Ensure to replace all variables related to datasets and annotations according to your specific setup. 
The hyperparameters and class names for testing are controled via [config.py](src/config.py).

# Technical Overview
The repository includes an [environment.yaml](environment.yaml) file containing all the necessary libraries. Key libraries used in this project include:

- Pytorch: For model training and inference.
- Scikit-learn: For metrics calculation.
- OpenSlide: For accessing WSIs.
- Other Libraries: numpy, pandas, matplotlib, cv2, seaborn.
