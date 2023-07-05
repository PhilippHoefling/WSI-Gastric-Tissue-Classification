
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model

if __name__ == "__main__":
    # Define dataset paths
    dataset_path = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed'
    # model_folder = ""
    #test_folder = "data_combined/test"
    #single_image_path = 'data_combined/test/scissors/scissors_1.jpg'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_new_transferlearning_model = False
    train_new_baseline_model = True
    test_existing_model = False
    #prediction_on_single_image = False
    #prediction_on_images = False
    #model_metrices = False
    #LIME_single_Image = False
    #activate_Augmentation = False

    #Set activate_Augmentation=False means training without data augmentation
    if train_new_baseline_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_path,tf_model=False)
        logger.info("Congratulations, training the baseline models was successful!")


#%%
