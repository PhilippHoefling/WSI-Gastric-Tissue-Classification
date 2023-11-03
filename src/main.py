
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model
from process import load_sort_data
from Tile_inference import tile_inference_regression, tile_inference_binary, pred_on_single_image
from process import plot_file_distribution
from WSI_Inference import TestOnSingleSlide
from VisionTransformer import trainVIT
from AntrumCorpusIntermediate import train_new_3model, print_3model_metrices
from BaseLine_Inflammed import train_new_inf_model
from auxiliaries import plot_loss_acc_curves

if __name__ == "__main__":
    # Define dataset paths
    dataset_Tissue_path = 'data/TissueTiles'
    dataset_Inflammed_path = 'data/InflamedTiles'
    model_folder = "models/TransferLearning_model_30102023_1009"
    test_Tissue_folder = "data/TissueTiles/test"
    test_Inf_folder = "data/InflamedTiles/test"
    tf_model ="imagenet"
    single_image_path = 'data/TissueTiles/test/corpus/47HE d-5_x-16540_y-3145_w-2560_h-2560_corpus.png'
    test_slidepath = 'C:/Users/phili//OneDrive - Otto-Friedrich-Universität Bamberg/DataSpell/xAIMasterThesis/data/WSIs/94HE.mrxs'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_model = False
    train_vit = False
    train_inf_model = True

    test_Tissue_Model = False
    test_Inf_Model = False
    testonWSI =  False
    test_3model = False
    prediction_on_image = False

    preprocess = False
    plot_data_distribution = False

    printLossCurves = False
    train_3model = False


    #model_metrices = False
    #activate_Augmentation = False



    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/DigPatTissue2/tiles",dataset_Tissue_path)
        logger.info("Congratulations, the preprocessing was successful!")

    if train_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_Tissue_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))


    if test_Tissue_Model:
        logger.info("Start testing the model..")
        tile_inference_regression(model_folder=model_folder, test_folder=test_Tissue_folder, safe_wrong_preds=True)

    if test_Inf_Model:
        logger.info("Start testing the model..")
        tile_inference_binary(model_folder=model_folder, test_folder=test_Inf_folder, safe_wrong_preds=True)    #%%

    if plot_data_distribution:
        logger.info("Start analyzing dataset..")
        plot_file_distribution(dataset_path=dataset_Tissue_path)   #%%

    if prediction_on_image:
        logger.info("Start prediction on single image...")
        pred_on_single_image(model_folder=model_folder,single_image_path=single_image_path)   #%%

    if testonWSI:
        logger.info("Start prediction on WSI...")
        TestOnSingleSlide(model_folder=model_folder, slidepath=test_slidepath)   #%%

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
        print_3model_metrices(model_folder=model_folder, test_folder=test_Tissue_folder)   #%%

#%%
