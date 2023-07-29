import numpy as np
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def get_model(model_folder: str):
    #get model from model folder
    onlyfiles = [f for f in listdir(model_folder) if isfile(join(model_folder, f))]

    model_path = model_folder / onlyfiles[1]
    results_path = model_folder / onlyfiles[2]
    hyperparameters_path = model_folder / onlyfiles[0]

    with open(model_path, "rb") as fid:
        classifier_model = pickle.load(fid)

    with open(results_path, "rb") as fid:
        results = pickle.load(fid)

    with open(hyperparameters_path, "rb") as fid:
        dict = pickle.load(fid)

    logger.info("Model and hyperparameters loaded!")
    logger.info("The model is trained with the following hyperparameters:")
    logger.info(dict)

    return classifier_model, results, dict

def plot_roc_curve(y_true, y_scores):
    """
    Plots the ROC curve for a binary classification model.

    Parameters:
        y_true (array-like): The true binary labels.
        y_scores (array-like): The predicted probabilities for the positive class.

    Returns:
        None (displays the ROC curve plot)
    """
    # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate the Area Under the Curve (AUC)
    auc_score = roc_auc_score(y_true, y_scores)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

