
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model
from process import load_sort_data
from evaluation import print_model_metrices, pred_on_single_image
from process import plot_file_distribution
from Tiling import TestonSlide
import torchvision
import torch

if __name__ == "__main__":
    # Define dataset paths
    dataset_path = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed'
    model_folder = "models/Baseline_model_11082023_1140"
    test_folder = "data/Processed/test"
    tf_model =""
    single_image_path = 'data/Processed/test/64HE d-5_x-2740_y-118040_w-2560_h-2560_antrum.png'
    test_slidepath = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/WSIs/12HE.mrxs'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_model = False
    test_existing_model = False
    preprocess = False
    plot_data_distribution = True
    prediction_on_image = False
    testonWSI =  False
    #model_metrices = False
    #activate_Augmentation = False



    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/DigPatFinal/tiles","C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed/")
        logger.info("Congratulations, the preprocessing was successful!")

    if train_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))


    if test_existing_model:
        logger.info("Start testing the model..")
        print_model_metrices(model_folder=model_folder, test_folder=test_folder)   #%%

    if plot_data_distribution:
        logger.info("Start analyzing dataset..")
        plot_file_distribution(dataset_path=dataset_path)   #%%

    if prediction_on_image:
        logger.info("Start prediction on single image...")
        pred_on_single_image(model_folder=model_folder,single_image_path=single_image_path)   #%%

    if testonWSI:
        logger.info("Start prediction on WSI...")
        TestonSlide(model_folder=model_folder, slidepath=test_slidepath)   #%%


#%%
