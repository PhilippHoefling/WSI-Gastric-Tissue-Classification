import numpy as np
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score, roc_curve, auc
import torch
import torchvision
from torchvision import transforms
import os
from pathlib import Path
from loguru import logger
from PIL import Image
from config import config_hyperparameter as cfg_hp

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



def pred_on_single_image(single_image_path, model_folder: str):
    '''
    Make predictions on a single images and plot the images with the prediction
    return: Nothing
    '''

    class_names = cfg_hp["class_names"]

    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)

    target_image= torchvision.io.read_image(str(single_image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255

    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
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
    # (using torch.softmax() for multi-class classification)
    _, predicted_idx = torch.max(target_image_pred, 1)

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    prediction_label = class_names[predicted_idx]

    image = Image.open(single_image_path)
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {prediction_label}" + "  Probabilities " + str(target_image_pred_probs[0].tolist()))

    plt.axis('off')
    plt.show()


def print_model_metrices(model_folder: str, test_folder: str):
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)
    image_path_list = list(Path(test_folder).glob("*/*.*"))
    class_names = cfg_hp["class_names"]
    accuracy = []
    probabilities = []
    predictions = []
    y_test = []

    for image_path in image_path_list:
        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255

        manual_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
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
            probabilities.append(torch.sigmoid(target_image_pred).item())
        # Convert logits -> prediction probabilities
        # (using torch.softmax() for multi-class classification)
        #target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_probs = torch.sigmoid(target_image_pred).round()


    # Convert prediction probabilities -> prediction labels
        target_image_pred_label = target_image_pred_probs.round()
        pred_class = class_names[int(target_image_pred_label)]
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
    print(probabilities)
    plot_roc_curve(model_folder=model_folder, y_true=y_test, y_scores=probabilities)


    #plot_roc_curve(y_test,target_image_pred_probs)
def plot_roc_curve(model_folder, y_true, y_scores, title='ROC Curve'):


    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    print(fpr)
    print(tpr)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig(model_folder + '/test_ROC_curve.png')
    plt.show()

