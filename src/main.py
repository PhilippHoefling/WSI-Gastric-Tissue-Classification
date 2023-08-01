
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model
from process import load_sort_data
from evaluation import print_model_metrices

if __name__ == "__main__":
    # Define dataset paths
    dataset_path = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed'
    model_folder = "models/Baseline_model_01082023_1428"
    test_folder = "data/Processed/test"
    tf_model =""
    #test_folder = "data_combined/test"
    #single_image_path = 'data_combined/test/scissors/scissors_1.jpg'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_model = False
    test_existing_model = True
    preprocess = False
    #prediction_on_images = False
    #model_metrices = False
    #activate_Augmentation = False



    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/DigPat2/tiles","C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed/")
        logger.info("Congratulations, training the preprocessing was successful!")

    if train_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))


    if test_existing_model:
        logger.info("Start testing the model..")
        print_model_metrices(model_folder=model_folder, test_folder=test_folder)   #%%
