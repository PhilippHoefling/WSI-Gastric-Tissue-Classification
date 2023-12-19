# Overview
This Repository contains the code of the master thesis of Philipp Höfling, written at the University of Bamberg, with the title "Development of an AI-based algorithm for the classification of gastric tissue in computational pathology".
This repository features prototype training pipelines, specifically designed for ResNet18 models, focusing on gastric mucosa classification.
The illustrated WSI test pipeline, developed in this work, enables trained ResNet18 models to classify whole slide images (WSIs), providing preliminary diagnoses and tissue heatmaps.
![PipelineResult](https://github.com/PhilippHoefling/WSI-Gastric-Tissue-Classification/assets/40239939/19b6c188-6f6f-4319-a1a3-65a57409be17)
The dataset and the final models are hosted at the the [Otto-Friedrich University Bamberg's chair of Explainable Machine Learning](https://www.uni-bamberg.de/xai/).
# Data
The datasets located at the university chair's file server. Please contact the university chair for access to the datasets and the project files.

# Technical Overview
The repository includes an [environment.yaml](environment.yaml) file containing all the necessary libraries. Key libraries used in this project include:

- Pytorch: For model training and inference.
- Scikit-learn: For metrics calculation.
- OpenSlide: For accessing WSIs.
- Other Libraries: numpy, pandas, matplotlib, cv2, seaborn.

# Environment Setup

This project uses a Conda environment to manage dependencies. To set up the environment, follow these steps:

1. **Install Conda**: If you don't have Conda installed, download and install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Clone the Repository**: If you haven't already, clone the repository to your local machine:
    ```bash
    git clone https://github.com/PhilippHoefling/WSI-Gastric-Tissue-Classification.git
    ```

3. **Navigate to the Project Directory**: Change to the project directory in your terminal:
    ```bash
    cd WSI-Gastric-Tissue-Classification
    ```

4. **Create the Conda Environment**: Use the `environment.yml` file to create a new Conda environment. Run the following command:
    ```bash
    conda env create -f environment.yml
    ```

5. **Activate the Environment**: Once the environment is created, activate it with:
    ```bash
    conda activate DigPat
    ```

6. **Verify the Environment**: Ensure that all the dependencies are correctly installed.

   
# Usage
All methods and pipelines are accessed via [main.py](src/main.py). 
Ensure to replace all variables related to datasets, models and annotations according to your specific setup. 
The hyperparameters and class names for testing are controled via [config.py](src/config.py).
  
# Contribution
Contributions to this repository are welcome. If you are interested in improving the pipelines or experimenting with the pipelines, please fork the repository.

# License
This project is licensed under the terms of the MIT license. For more details, see [License](LICENSE)
