# Data Preprocessing

# ## Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from config import config_hyperparameter as cfg_hp
import shutil
import csv


def loading(folder_name: str):
    direc = os.path.join(
        folder_name)
    return os.listdir(direc), len(os.listdir(direc))


def load_sort_data(src_dir: str, dst_dir: str, use_split: bool):
    # check if src dir exists
    if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
        print(f"Source path '{src_dir}' is not a valid directory.")

    # Set the random seed
    np.random.seed(cfg_hp["seed"])

    labels = 'corpus', 'antrum', 'intermediate'
    #labels = '_inflamed','_noninflamed'

    folders, num_folders = loading(src_dir)
    npfol = np.array(folders)
    np.random.shuffle(npfol)

    #use fixed split
    if use_split:
        #fixed split used by Tom and Philipp
        train = ['61HE','28BHE','84HE','80HE','24CHE','15SHE','10KHE','14BHE','11SHE'
            ,'45HE','1SHE','27CHE','87HE','46BHE','4BHE','43HE','35CHE','37HE','11HE'
            ,'8KHE','102HE','56HE','100HE','39HE','31BHE','9BHE'
            ,'93HE','79HE','37CHE','1HE','1CHE','10SHE','26BHE','15KHE','62HE','30CHE'
            ,'7CHE','34HE','91HE','64HE','75HE','69HE','23BHE','44BHE','16BHE','21CHE'
            ,'40CHE','65HE','7HE','34BHE','98HE','8SHE','90HE','85HE','26HE','73HE'
            ,'22CHE','48BHE','5HE','33BHE','12CHE!','22HE','57HE','2HE','16HE','41BHE'
            ,'8BHE','10HE','101HE','43BHE','30BHE','3BHE','54BHE','1KHE','17HE'
            ,'39BHE','33CHE','50BHE','26CHE','11BHE','7KHE','12SHE','37BHE','17KHE'
            ,'68HE','3HE','27BHE','32CHE','94HE','16SHE','32HE','40HE','17BHE','6CHE'
            ,'31HE','49BHE','49HE','1BHE','12HE','14CHE','82HE','14SHE','30HE','7BHE'
            ,'13KHE','19BHE','58BHE','27HE','5CHE','2SHE','86HE','4CHE','9SHE','45BHE'
            ,'38BHE','29CHE','42BHE','58HE','42CHE','72HE','23HE','88HE','43CHE'
            ,'20CHE','44HE','24HE','31CHE','2BHE','56BHE','97HE','74HE','12BHE','96HE'
            ,'55BHE','50HE','13HE','53BHE','48HE','55HE','104HE','71HE','54HE','10CHE'
            ,'11BHE','105HE','14HE','59BHE','21HE','35BHE','9BHE'
            ,'46HE','9HE','16KHE','5SHE','3SHE','16CHE','52BHE','28CHE','11CHE!'
            ,'29BHE','81HE','92HE','3KHE','10BHE','59HE','13CHE!','24BHE','8HE'
            ,'41CHE','13BHE','47HE','19HE','34CHE','67HE','99HE','63HE','32BHE','6SHE'
            ,'10BHE','13SHE','25CHE','9CHE','6BHE','4SHE','38HE'
            ,'51BHE','4KHE','57BHE','8CHE','103HE','52HE','60HE']
        validate = ['33HE','18BHE','28HE','21BHE','42HE','7SHE','13SHE'
            ,'39CHE','70HE','60BHE','18CHE','53HE','25BHE','36HE','20HE','76HE'
            ,'14KHE','5BHE','78HE','83HE','15HE','41HE','38CHE','95HE']
        test = ['25HE','15BHE','36CHE','6HE','2CHE','77HE','3CHE','5KHE','40BHE','15CHE'
            ,'9KHE','29HE','51HE','35HE','6KHE','2KHE','66HE','47BHE','19CHE','11KHE'
            ,'20BHE','12KHE','18HE','22BHE','23CHE']
    else:
        #or generate new Split and save it
        npfol = np.array(folders)
        np.random.shuffle(npfol)
        train, validate, test = np.split(npfol, [int(.8 * len(npfol)), int(.9 * len(npfol))])
        # Open the CSV file in write mode
        with open("data_split.csv", 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            # Write the string as a single-row CSV entry
            csv_writer.writerow(["train: " + str(train)])
            csv_writer.writerow(["val: " + str(validate)])
            csv_writer.writerow(["test: " + str(test)])
    # split dictionary
    splits = {
        "train": train,
        "validate": validate,
        "test": test
    }

    # load and sort all tiles from tileexporter in the according folder
    for folder in folders:
        dir_folder = src_dir + "/" + folder
        subfolders, num_folders = loading(dir_folder)
        for subfolder in subfolders:
            dir_subsubfolder = dir_folder + "/" + subfolder
            images, num_images = loading(dir_subsubfolder)
            for image in images:
                dir_image = dir_subsubfolder + "/" + image
                # check if the image contains more than 5% tissue
                for label in labels:
                    if label in image and is_white_or_grey_png(dir_image):
                        if folder in splits["train"]:
                            shutil.copyfile(dir_image, dst_dir + "/train/" + label.replace("_", "") + "/" + image)
                        if folder in splits["validate"]:
                            shutil.copyfile(dir_image, dst_dir + "/val/" + label.replace("_", "") + "/" + image)
                        if folder in splits["test"]:
                            shutil.copyfile(dir_image, dst_dir + "/test/" + label.replace("_", "") + "/" + image)


def plot_file_distribution(dataset_path):
    folders, num_folders = loading(dataset_path)

    # Set the global font size
    plt.rcParams.update({'font.size': 14})

    classes = cfg_hp["class_names"]
    datasets = ["train", "val", "test"]

    class_counts = {dataset: {cls: 0 for cls in classes} for dataset in datasets}

    for dataset in datasets:
        for cls in classes:
            class_folder = os.path.join(dataset_path, dataset, cls)
            if os.path.isdir(class_folder):
                class_counts[dataset][cls] = len(os.listdir(class_folder))

    num_datasets = len(datasets)
    num_classes = len(classes)
    bar_width = 0.8 / num_classes  # Adjusted bar width
    index = np.arange(num_datasets)

    colors = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cls in enumerate(classes):
        counts = [class_counts[dataset][cls] for dataset in datasets]
        bar_positions = index + i * bar_width
        ax.bar(bar_positions, counts, bar_width, label=cls, color=colors[i])

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution')
    ax.set_xticks(index + bar_width * num_classes / 2)  # Center the x-axis ticks
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)  # Add grid lines
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)  # Save the figure with high resolution
    plt.show()


def is_white_or_grey_png(image_path, threshold=0.90):
    try:
        image = Image.open(image_path)
    except IOError:
        raise ValueError("Unable to open the image file")

    # Convert the image to grayscale for easy white/grey detection
    grayscale_image = image.convert("L")

    # Get the pixel data from the image
    pixels = grayscale_image.load()

    # Get the image size
    width, height = image.size

    # Count the number of white/grey pixels in the image
    white_or_grey_pixel_count = sum(pixels[x, y] >= 200 for x in range(width) for y in range(height))

    # Calculate the percentage of white/grey pixels in the image
    white_or_grey_percentage = white_or_grey_pixel_count / float(width * height)

    # Check if the white/grey percentage is above the threshold
    return white_or_grey_percentage < threshold

def get_WSI_aggregation_data(src_dir: str):
    # check if src dir exists
    if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
        print(f"Source path '{src_dir}' is not a valid directory.")

    # Set the random seed
    np.random.seed(cfg_hp["seed"])

    labels = 'corpus', 'antrum', 'intermediate'
    #labels = '_inflamed','_noninflamed'

    folders, num_folders = loading(src_dir)
    npfol = np.array(folders)
    np.random.shuffle(npfol)


    #fixed split used by Tom and Philipp
    train = ['61HE','28BHE','84HE','80HE','24CHE','15SHE','10KHE','14BHE','11SHE'
        ,'45HE','1SHE','27CHE','87HE','46BHE','4BHE','43HE','35CHE','37HE','11HE'
        ,'8KHE','102HE','56HE','100HE','39HE','31BHE','9BHE'
        ,'93HE','79HE','37CHE','1HE','1CHE','10SHE','26BHE','15KHE','62HE','30CHE'
        ,'7CHE','34HE','91HE','64HE','75HE','69HE','23BHE','44BHE','16BHE','21CHE'
        ,'40CHE','65HE','7HE','34BHE','98HE','8SHE','90HE','85HE','26HE','73HE'
        ,'22CHE','48BHE','5HE','33BHE','12CHE!','22HE','57HE','2HE','16HE','41BHE'
        ,'8BHE','10HE','101HE','43BHE','30BHE','3BHE','54BHE','1KHE','17HE'
        ,'39BHE','33CHE','50BHE','26CHE','11BHE','7KHE','12SHE','37BHE','17KHE'
        ,'68HE','3HE','27BHE','32CHE','94HE','16SHE','32HE','40HE','17BHE','6CHE'
        ,'31HE','49BHE','49HE','1BHE','12HE','14CHE','82HE','14SHE','30HE','7BHE'
        ,'13KHE','19BHE','58BHE','27HE','5CHE','2SHE','86HE','4CHE','9SHE','45BHE'
        ,'38BHE','29CHE','42BHE','58HE','42CHE','72HE','23HE','88HE','43CHE'
        ,'20CHE','44HE','24HE','31CHE','2BHE','56BHE','97HE','74HE','12BHE','96HE'
        ,'55BHE','50HE','13HE','53BHE','48HE','55HE','104HE','71HE','54HE','10CHE'
        ,'11BHE','105HE','14HE','59BHE','21HE','35BHE','9BHE'
        ,'46HE','9HE','16KHE','5SHE','3SHE','16CHE','52BHE','28CHE','11CHE!'
        ,'29BHE','81HE','92HE','3KHE','10BHE','59HE','13CHE!','24BHE','8HE'
        ,'41CHE','13BHE','47HE','19HE','34CHE','67HE','99HE','63HE','32BHE','6SHE'
        ,'10BHE','13SHE','25CHE','9CHE','6BHE','4SHE','38HE'
        ,'51BHE','4KHE','57BHE','8CHE','103HE','52HE','60HE']
    validate = ['33HE','18BHE','28HE','21BHE','42HE','7SHE','13SHE'
        ,'39CHE','70HE','60BHE','18CHE','53HE','25BHE','36HE','20HE','76HE'
        ,'14KHE','5BHE','78HE','83HE','15HE','41HE','38CHE','95HE']
    test = ['25HE','15BHE','36CHE','6HE','2CHE','77HE','3CHE','5KHE','40BHE','15CHE'
        ,'9KHE','29HE','51HE','35HE','6KHE','2KHE','66HE','47BHE','19CHE','11KHE'
        ,'20BHE','12KHE','18HE','22BHE','23CHE']





    # split dictionary
    splits = {
        "train": train,
        "validate": validate,
        "test": test
    }

    # Initialize dictionaries
    train_WSIs = {}
    val_WSIs = {}
    test_WSIs = {}

    # load and sort all tiles from tileexporter in the according folder
    for folder in folders:
        dir_folder = src_dir + "/" + folder
        subfolders, num_folders = loading(dir_folder)
        for subfolder in subfolders:
            dir_subsubfolder = dir_folder + "/" + subfolder
            images, num_images = loading(dir_subsubfolder)
            for image in images:
                dir_image = dir_subsubfolder + "/" + image
                # check if the image contains more than 5% tissue
                for label in labels:
                    clean_label = label.replace("_", "")
                    if label in image and is_white_or_grey_png(dir_image):
                        if folder in splits["train"]:
                            add_label_to_dict(train_WSIs, folder, clean_label)
                        if folder in splits["validate"]:
                            add_label_to_dict(val_WSIs, folder, clean_label)
                        if folder in splits["test"]:
                            add_label_to_dict(test_WSIs, folder, clean_label)


    with open("WSI_Path_Tissue.csv", 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        # Write the string as a single-row CSV entry
        csv_writer.writerow(["train: " + str(train_WSIs)])
        csv_writer.writerow(["val: " + str(val_WSIs)])
        csv_writer.writerow(["test: " + str(test_WSIs)])

        # Function to remove duplicates by converting lists to tuples
def add_label_to_dict(wsi_dict, wsi, label):
    if wsi not in wsi_dict:
        wsi_dict[wsi] = []
    if label not in wsi_dict[wsi]:
        wsi_dict[wsi].append(label)