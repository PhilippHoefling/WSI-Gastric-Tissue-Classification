import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score, roc_curve, auc
import torch
import torchvision
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from config import config_hyperparameter as cfg_hp
import csv
from auxiliaries import get_model


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
        target_image_pred = torch.sigmoid(target_image_pred)
        print(target_image_pred)
    # )
    _, predicted_idx = torch.max(target_image_pred, 1)


    target_image_pred_probs = target_image_pred.item()
    prediction_label = class_names[predicted_idx]

    image = Image.open(single_image_path)
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {prediction_label}" + "  Probabilities " + str(target_image_pred_probs))

    plt.axis('off')
    plt.show()
def tile_inference_binary(model_folder: str, test_folder: str, safe_wrong_preds: bool):
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)
    image_path_list = list(Path(test_folder).glob("*/*.*"))
    class_names = cfg_hp["class_names"]
    accuracy = []
    predictions = []
    y_test = []
    false_pred_path = []
    prob_distibution = []
    probabilities = []

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
            target_image_pred_probs = torch.sigmoid(target_image_pred).round()

            probabilities.append(torch.sigmoid(target_image_pred).item())

        # Convert prediction probabilities -> prediction labels
        target_image_pred_label = target_image_pred_probs.round()
        pred_class = class_names[int(target_image_pred_label)]
        true_class = image_path.parts[-2]

        prob_distibution.append([true_class , torch.sigmoid(target_image_pred)])

        predictions.append(class_names.index(pred_class))
        y_test.append(class_names.index(true_class))

        if pred_class == true_class:
            accuracy.append(1)
        else:
            accuracy.append(0)
            false_pred_path.append(str(image_path))


    #safe wrong predictions
    if safe_wrong_preds:
        with open("Val_Classification_Errors.csv", 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            # Write the string as a single-row CSV entry

            csv_writer.writerow( false_pred_path)


    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=class_names, cmap='Blues',
                                            colorbar=False)
    plt.savefig(model_folder + '/test_confusion_matrix.png')
    plt.show()

    print("Accuracy on test set: " + str(sum(accuracy) / len(accuracy) * 100) + " %")
    print("Precision on test set " + str(precision_score(y_test, predictions, average='binary')))
    print("Sensitivity on test set " + str(recall_score(y_test, predictions, average='binary')))
    print("F1 Score on test set " + str(f1_score(y_test, predictions, average='binary')))
    # print("Log-Loss on test set " + str(log_loss(y_test, predictions)))


    plot_roc_curve(model_folder=model_folder, y_true=y_test, y_scores=probabilities)

    plot_prob_distribution(model_folder=model_folder,  prob_distibution=prob_distibution)

    print(y_test)
    print(predictions)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=class_names, cmap='Blues',
                                            colorbar=False)
    plt.rcParams.update({'font.size': 14})
    plt.savefig(model_folder + '/test_confusion_matrix.png')
    plt.show()

    print("Accuracy on test set: " + str(sum(accuracy) / len(accuracy) * 100) + " %")
    print("Precision on test set " + str(precision_score(y_test, predictions, average='macro')))
    print("Recall on test set " + str(recall_score(y_test, predictions, average='macro')))
    print("F1 Score on test set " + str(f1_score(y_test, predictions, average='macro')))
    # print("Log-Loss on test set " + str(log_loss(y_test, predictions)))


    #plot_roc_curve(model_folder=model_folder, y_true=y_test, y_scores=probabilities)

    plot_prob_distribution(model_folder=model_folder,  prob_distibution=prob_distibution)

    #plot_roc_curve(y_test,target_image_pred_probs)
def plot_roc_curve(model_folder, y_true, y_scores, title='ROC Curve'):


    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.rcParams.update({'font.size': 14})
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

def plot_prob_distribution(model_folder, prob_distibution):
    # Find unique Classes
    unique_classes = list(set([entry[0] for entry in prob_distibution]))

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    for true_class in unique_classes:
        target_probs = [entry[1] for entry in prob_distibution if entry[0] == true_class]
        # Since target_image_pred_probs is a tensor, we need to convert it to a numpy array and flatten it.
        flat_probs = [prob.item() for sublist in target_probs for prob in sublist.cpu().numpy().flatten()]
        plt.hist(flat_probs, bins=30, alpha=0.6, label=f"Class {true_class}")

    plt.title("Distribution of Predicted Probabilities for Each True Class")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(model_folder + '/Test_Prob_Distr.png')
    plt.show()


#%%
