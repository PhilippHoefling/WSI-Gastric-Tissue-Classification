
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model
from process import load_sort_data
from evaluation import print_model_metrices, pred_on_single_image
from process import plot_file_distribution

if __name__ == "__main__":
    # Define dataset paths
    dataset_path = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed'
    model_folder = "models/TransferLearning_model_04082023_1018"
    test_folder = "data/Processed/test"
    tf_model ="imagenet"
    #test_folder = "data_combined/test"
    single_image_path = 'data/Processed/test/....'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_model = False
    test_existing_model = True
    preprocess = False
    plot_data_distribution = False
    prediction_on_images = False
    #model_metrices = False
    #activate_Augmentation = False



    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/DigPat2/tiles","C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed/")
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

    if prediction_on_images:
        logger.info("Start prediction on single image...")
        pred_on_single_image(model_folder=model_folder,image_path=single_image_path)   #%%

#%%
