import pandas as pd

from sklearn.metrics import (ConfusionMatrixDisplay, precision_score, f1_score, recall_score,
                             roc_curve, auc, accuracy_score, confusion_matrix, cohen_kappa_score)
from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score, roc_curve, auc
import torch
import torchvision
from torchvision import transforms
from auxiliaries import get_model
from config import config_hyperparameter as cfg_hp

def evaluate_Experiment_Tiles(csvpathInflammation: str, csvpathTissue: str):

    #print("Inflammed Tiles Experiment Analysis")
    #evaluate_Inflammed_Tiles(csvpathInflammation=csvpathInflammation)
    #print("Tissue Tiles Experiment Analysis")
    #evaluate_Tissue_Tiles(csvpathTissue=csvpathTissue)
    evaluate_Tissue_Tiles_ROC(csvpathTissue="results experiment/Experiment Human Performance Tissue.csvAI - Without Intermediate.csv")

    print("Inflamed AI evaluation")
    evaluate_Inflammed_Tiles_AI(csvpathInflammation=csvpathInflammation)

def evaluate_Inflammed_Tiles(csvpathInflammation: str):
    # Load the CSV into a DataFrame
    df_csv = pd.read_csv(csvpathInflammation, sep=';')



    # Compute metrice for each pathologist
    for pathologist in ['Pathologist 1', 'Pathologist 2']:

        accuracy = accuracy_score(df_csv['Ground_Truth'], df_csv[pathologist])
        precision = precision_score(df_csv['Ground_Truth'], df_csv[pathologist])
        recall = recall_score(df_csv['Ground_Truth'], df_csv[pathologist])
        f1 = f1_score(df_csv['Ground_Truth'], df_csv[pathologist])

        print(f"accuracy of  {pathologist} :{accuracy}")
        print(f"precision of  {pathologist} :{precision}")
        print(f"recall of  {pathologist} :{recall}")
        print(f"f1 - score of  {pathologist} :{f1}")

        fpr, tpr, _ = roc_curve(df_csv['Ground_Truth'], df_csv[pathologist])
        roc_auc = auc(fpr, tpr)

        # Set the global font size
        plt.rcParams.update({'font.size': 14})

        # Plotting the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {pathologist}')
        plt.legend(loc="lower right")
        plt.savefig( f"results experiment/ROC_Inflammed_Tiles_{pathologist}")
        plt.show()

        # Displaying the Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix(df_csv['Ground_Truth'], df_csv[pathologist]), display_labels=['non inflamed', 'inflamed'])
        disp.plot(cmap='Blues', values_format='d')
        # Adjust layout
        plt.tight_layout()
        plt.savefig( f"results experiment/Confusion_Matrice_Inflammed_{pathologist}")
        plt.show()

    # Calculate Cohens Kappa between Pathologists
    kappa_between_pathologists = cohen_kappa_score(df_csv['Pathologist 1'], df_csv['Pathologist 2'])
    print(f"Cohen's Kappa between Pathologists: {kappa_between_pathologists}")

def evaluate_Tissue_Tiles(csvpathTissue: str):
    # Load the CSV into a DataFrame
    df_csv = pd.read_csv(csvpathTissue, sep=',')
    print(df_csv.columns)

    # List of classes
    classes = ['C', 'A', 'I']

    # Binarize the output for ROC curve calculation
    y_true = label_binarize(df_csv['Ground_Truth'], classes=classes)

    # Compute metrics for each pathologist
    for pathologist in ['Pathologist 1', 'Pathologist 2','AI_model']:
        y_pred = label_binarize(df_csv[pathologist], classes=classes)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"accuracy of  {pathologist} :{accuracy}")
        print(f"precision of  {pathologist} :{precision}")
        print(f"recall of  {pathologist} :{recall}")
        print(f"f1 - score of  {pathologist} :{f1}")

        # Set the global font size
        plt.rcParams.update({'font.size': 14})

        # Displaying the Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix(df_csv['Ground_Truth'], df_csv[pathologist], labels=classes), display_labels=classes)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix for {pathologist}')
        plt.savefig(f"results experiment/Confusion_Tissue_Matrice_{pathologist}")
        plt.show()

    # Calculate Cohen's Kappa between Pathologists
    kappa_between_pathologists = cohen_kappa_score(df_csv['Pathologist 1'], df_csv['Pathologist 2'])
    print(f"Cohen's Kappa between Pathologists: {kappa_between_pathologists}")
    # Calculate Cohen's Kappa between each Pathologist and the AI model
    kappa_pathologist1_ai = cohen_kappa_score(df_csv['Pathologist 1'], df_csv['AI_model'])
    kappa_pathologist2_ai = cohen_kappa_score(df_csv['Pathologist 2'], df_csv['AI_model'])
    print(f"Cohen's Kappa between Pathologist 1 and AI model: {kappa_pathologist1_ai}")
    print(f"Cohen's Kappa between Pathologist 2 and AI model: {kappa_pathologist2_ai}")
def evaluate_Tissue_Tiles_ROC(csvpathTissue: str):
    # Load the CSV into a DataFrame
    df_csv = pd.read_csv(csvpathTissue, sep=',')
    print(df_csv.columns)

    # List of classes
    classes = ['C', 'A']

    # Binarize the output for ROC curve calculation
    y_true = label_binarize(df_csv['Ground_Truth'], classes=classes)

    # Compute metrics for each pathologist
    for pathologist in ['Pathologist 1', 'Pathologist 2','AI_model']:
        y_pred = label_binarize(df_csv[pathologist], classes=classes)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy of {pathologist}: {accuracy}")
        print(f"Precision of {pathologist}: {precision}")
        print(f"Recall of {pathologist}: {recall}")
        print(f"F1-Score of {pathologist}: {f1}")

        # Calculate ROC curve and ROC area for each pathologist
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc = auc(fpr, tpr)

        print(f"ROC AUC of {pathologist}: {roc_auc}")

        # Set the global font size
        plt.rcParams.update({'font.size': 14})

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {pathologist}')
        plt.legend(loc="lower right")
        plt.savefig(f"results experiment/ROC_Tissue_Tiles_{pathologist}.png")
        plt.show()
def AI_Tile_prediction_for_Experiment(csv_file, folder_tiles, model_folder: str):
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=';')

    # Initialize lists to store filenames and model predictions
    filenames = []
    model_predictions = []

    class_names = cfg_hp["class_names"]


    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)


    # Iterate through each file name in the DataFrame
    for file_name in df['Tile_file_name']:
        # Add the file name to the filenames list
        filenames.append(file_name)
        #Built path
        tilepath = folder_tiles + '/' + file_name

        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(tilepath)).type(torch.float32)

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

        # Convert prediction probabilities -> prediction labels
        target_image_pred_label = target_image_pred_probs.round()
        #pred_class = class_names[int(target_image_pred_label)]


        # Get the model prediction for the file
        #prediction = class_names.index(pred_class)

        # Map 0 to 'A' and 1 to 'C'
        pred_class = 'A' if target_image_pred_label == 0 else 'C'

        model_predictions.append(pred_class)

    # Add the filenames and model predictions as new columns in the DataFrame
    #df['Filename'] = filenames
    df['AI_model'] = model_predictions

    # Save the updated DataFrame to a new CSV file
    df.to_csv(path_or_buf= csv_file +"AI.csv", index=False)

def evaluate_Inflammed_Tiles_AI(csvpathInflammation: str):
    # Load the CSV into a DataFrame
    df_csv = pd.read_csv(csvpathInflammation, sep=',')

    # Metrics for AI model
    evaluator = 'AI_model'
    accuracy = accuracy_score(df_csv['Ground_Truth'], df_csv[evaluator])
    precision = precision_score(df_csv['Ground_Truth'], df_csv[evaluator])
    recall = recall_score(df_csv['Ground_Truth'], df_csv[evaluator])
    f1 = f1_score(df_csv['Ground_Truth'], df_csv[evaluator])

    print(f"Accuracy of {evaluator}: {accuracy}")
    print(f"Precision of {evaluator}: {precision}")
    print(f"Recall of {evaluator}: {recall}")
    print(f"F1-Score of {evaluator}: {f1}")

    fpr, tpr, _ = roc_curve(df_csv['Ground_Truth'], df_csv[evaluator])
    roc_auc = auc(fpr, tpr)

    # Set the global font size
    plt.rcParams.update({'font.size': 14})

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {evaluator}')
    plt.legend(loc="lower right")
    plt.savefig(f"results experiment/ROC_Inflammed_Tiles_{evaluator}")
    plt.show()

    # Displaying the Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix(df_csv['Ground_Truth'], df_csv[evaluator]), display_labels=['non inflamed', 'inflamed'])
    disp.plot(cmap='Blues', values_format='d')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"results experiment/Confusion_Matrice_Inflammed_{evaluator}")
    plt.show()

    # Calculate Cohen's Kappa between each pathologist and the AI model
    for pathologist in ['Pathologist 1', 'Pathologist 2']:
        kappa = cohen_kappa_score(df_csv[pathologist], df_csv['AI_model'])
        print(f"Cohen's Kappa between {pathologist} and AI model: {kappa}")
# Example usage
# process_files('path_to_your_csv.csv', your_model_object)
#%%
