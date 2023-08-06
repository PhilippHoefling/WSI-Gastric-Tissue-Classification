import numpy as np
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score, roc_curve, roc_auc_score
import torch
import torchvision
from torchvision import transforms
import os
from pathlib import Path
from loguru import logger

def get_model(model_folder: str):
    # get model from model folder
    onlyfiles = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
    model_folder = Path(model_folder)


    hyperparameters_path = model_folder.joinpath(onlyfiles[0])
    model_path = model_folder.joinpath(onlyfiles[1])
    results_path = model_folder.joinpath(onlyfiles[2])
    summary_path = model_folder.joinpath(onlyfiles[3])

    with open(model_path, "rb") as fid:
        classifier_model = pickle.load(fid)

    with open(results_path, "rb") as fid:
        results = pickle.load(fid)

    with open(hyperparameters_path, "rb") as fid:
        dict = pickle.load(fid)

    with open(summary_path, "rb") as fid:
        summary = pickle.load(fid)

    logger.info("Model and hyperparameters loaded!")
    logger.info("The model is trained with the following hyperparameters:")
    logger.info(dict)

    return classifier_model, results, dict, summary

def pred_on_single_image(image_path:str, model_folder:str):
    class_names = ['corpus','antrum']

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()

    trained_model, model_results, dict_hyperparameters = get_model(Path(model_folder))


    pred_and_plot_image(trained_model, image_path, class_names, auto_transforms)
def print_model_metrices(model_folder: str, test_folder: str):
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)
    image_path_list = list(Path(test_folder).glob("*/*.*"))
    class_names = ['corpus', 'antrum']
    accuracy = []
    predictions = []
    y_test = []

    for image_path in image_path_list:
        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255

        manual_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Transform if necessary
        target_image = manual_transforms(target_image)

        # Turn on model evaluation mode and inference mode
        trained_model.eval()
        with torch.inference_mode():
            # Add an extra dimension to the image
            target_image = target_image.unsqueeze(dim=0)

            # Make a prediction on image with an extra dimension
            target_image_pred = trained_model(target_image.cuda())

        # Convert logits -> prediction probabilities
        # (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        pred_class = class_names[target_image_pred_label.item()]
        true_class = image_path.parts[3]
        predictions.append(target_image_pred_label.item())
        y_test.append(class_names.index(true_class))
        if pred_class == true_class:
            accuracy.append(1)
        else:
            accuracy.append(0)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=class_names, cmap='Blues',
                                            colorbar=False)
    plt.savefig(model_folder + '/test_confusion_matrix.png')
    plt.show()

    print("Accuracy on test set: " + str(sum(accuracy) / len(accuracy) * 100) + " %")
    print("Precision on test set " + str(precision_score(y_test, predictions, average='macro')))
    print("Recall on test set " + str(recall_score(y_test, predictions, average='macro')))
    print("F1 Score on test set " + str(f1_score(y_test, predictions, average='macro')))
    # print("Log-Loss on test set " + str(log_loss(y_test, predictions)))

    plot_roc_curve(y_test,predictions)
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
    print("AUC Score on test set  " + str(auc_score))
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
