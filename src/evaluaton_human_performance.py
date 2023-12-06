import pandas as pd

from sklearn.metrics import (ConfusionMatrixDisplay, precision_score, f1_score, recall_score,
                             roc_curve, auc, accuracy_score, confusion_matrix, cohen_kappa_score)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt

def evaluate_Experiment_Tiles(csvpathInflammation: str, csvpathTissue: str):

    print("Inflammed Tiles Experiment Analysis")
    evaluate_Inflammed_Tiles(csvpathInflammation=csvpathInflammation)
    print("Tissue Tiles Experiment Analysis")
    evaluate_Tissue_Tiles(csvpathTissue=csvpathTissue)
    evaluate_Tissue_Tiles_ROC(csvpathTissue="results experiment/Experiment Human Performance Tissue - Without Intermediate.csv")
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
    df_csv = pd.read_csv(csvpathTissue, sep=';')
    print(df_csv.columns)

    # List of classes
    classes = ['C', 'A', 'I']

    # Binarize the output for ROC curve calculation
    y_true = label_binarize(df_csv['Ground_Truth'], classes=classes)

    # Compute metrics for each pathologist
    for pathologist in ['Pathologist 1', 'Pathologist 2']:
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

def evaluate_Tissue_Tiles_ROC(csvpathTissue: str):
    # Load the CSV into a DataFrame
    df_csv = pd.read_csv(csvpathTissue, sep=';')
    print(df_csv.columns)

    # List of classes
    classes = ['C', 'A']

    # Binarize the output for ROC curve calculation
    y_true = label_binarize(df_csv['Ground_Truth'], classes=classes)

    # Compute metrics for each pathologist
    for pathologist in ['Pathologist 1', 'Pathologist 2']:
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

#%%
