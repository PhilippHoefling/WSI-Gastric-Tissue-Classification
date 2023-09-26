
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model, plot_loss_acc_curves
from process import load_sort_data
from evaluation import print_model_metrices, pred_on_single_image
from process import plot_file_distribution
from Tiling import TestonSlide
from VisionTransformer import trainVIT
from AntrumCorpusIntermediate import train_new_3model, print_3model_metrices
from BaseLine_Inflammed import train_new_inf_model
import torchvision
import torch

if __name__ == "__main__":
    # Define dataset paths
    dataset_Tissue_path = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/TissueTiles'
    dataset_Inflammed_path = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/InflamedTiles'
    model_folder = "models/TransferLearning_model_24092023_1356"
    test_folder = "data/TissueTiles/test"
    tf_model ="imagenet"
    single_image_path = 'data/TissueTiles/test/corpus/47HE d-5_x-16540_y-3145_w-2560_h-2560_corpus.png'
    test_slidepath = 'C:/Users/phili/DataspellProjects/xAIMasterThesis/data/WSIs/7CHE.mrxs'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_model = False
    train_vit = True
    test_existing_model = False
    preprocess = False
    plot_data_distribution = False
    prediction_on_image = False
    testonWSI =  False
    printLossCurves = False
    train_3model = False
    train_inf_model = False
    test_3model = False
    #model_metrices = False
    #activate_Augmentation = False



    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/DigPatInflammed/tiles",dataset_Inflammed_path)
        logger.info("Congratulations, the preprocessing was successful!")

    if train_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_Tissue_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))


    if test_existing_model:
        logger.info("Start testing the model..")
        print_model_metrices(model_folder=model_folder, test_folder=test_folder, safe_wrong_preds=True)   #%%

    if plot_data_distribution:
        logger.info("Start analyzing dataset..")
        plot_file_distribution(dataset_path=dataset_Inflammed_path)   #%%

    if prediction_on_image:
        logger.info("Start prediction on single image...")
        pred_on_single_image(model_folder=model_folder,single_image_path=single_image_path)   #%%

    if testonWSI:
        logger.info("Start prediction on WSI...")
        TestonSlide(model_folder=model_folder, slidepath=test_slidepath)   #%%

    if printLossCurves:
        logger.info("Start printing loss/accuracy curves...")
        plot_loss_acc_curves(model_folder=model_folder)  #%%
    if train_vit:
        logger.info("Start training Visual Transformer...")
        model_folder = trainVIT(dataset_path=dataset_Inflammed_path)
        logger.info("Congratulations, training your vision transformer was successful!" + str(model_folder))

    if train_3model:
        model_folder = train_new_3model(dataset_path=dataset_Tissue_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if train_inf_model:
        model_folder = train_new_inf_model(dataset_path=dataset_Inflammed_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if test_3model:
        logger.info("Start testing the model..")
        print_3model_metrices(model_folder=model_folder, test_folder=test_folder)   #%%

#%%
