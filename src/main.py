
from loguru import logger
from BaseLine_AntrumCorpus import train_new_model
from process import load_sort_data
from Tile_inference import tile_inference_regression, tile_inference_binary, pred_on_single_image
from process import plot_file_distribution
from WSI_Inference import TestOnSingleSlide, TestOnSlides
from VisionTransformer import trainVIT
from AntrumCorpusIntermediate import train_new_3model, print_3model_metrices
from BaseLine_Inflammed import train_new_inf_model
from auxiliaries import plot_loss_acc_curves
from EfficientNetB5 import train_new_EFFICIENTNET
from evaluaton_human_performance import evaluate_Experiment_Tiles
import time

if __name__ == "__main__":
    # Define dataset paths
    dataset_Tissue_path = 'data/TissueTiles'
    dataset_Tom_Tissue = 'data/TissueTilesReworked'
    dataset_Inflammed_path = 'data/InflamedTiles'
    dataset_Inf_tom_path = 'data/InflamedTilesReworked'

    model_folder_inf = "models/TransferLearning_model_20112023_0854"
    model_folder_tissue = "models/TransferLearning_model_19092023_2040"

    test_Tissue_folder = "data/TissueTiles/test"
    test_Inf_folder = "data/InflamedTiles/test"
    test_Tom_Inf_folder = "data/InflamedTilesReworked/test"

    tf_model ="imagenet"
    single_image_path = 'data/InflamedTiles/test/inflamed/7CHE d-5_x-58735_y-137940_w-2560_h-2560_inflamed.png'
    test_slidepath = 'E:/Scans/Non-Inflamed/72HE.mrxs'

    # Set parameter for testing
    #num_images = 6

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_model = False
    train_vit = False
    train_inf_model = False
    train_effNet= False
    train_2_class_model = False

    test_Tissue_Model = False
    test_Inf_Model = False
    testonWSI =  False
    testonAllWSIS = False
    test_3model = True
    prediction_on_image = False

    preprocess = False
    plot_data_distribution = False

    printLossCurves = True


    #Evaluation of Experiment results
    evaluate_experiment_tile_level = False



    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/TomGastric/tiles",dataset_Tom_Tissue, use_split=True)
        logger.info("Congratulations, the preprocessing was successful!")

    if train_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_Tissue_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))


    if test_Tissue_Model:
        logger.info("Start testing the model..")
        tile_inference_binary(model_folder=model_folder_tissue, test_folder=test_Tissue_folder, safe_wrong_preds=False)

    if test_Inf_Model:
        logger.info("Start testing the model..")
        tile_inference_binary(model_folder=model_folder_inf, test_folder=test_Tom_Inf_folder, safe_wrong_preds=False)    #%%

    if plot_data_distribution:
        logger.info("Start analyzing dataset..")
        plot_file_distribution(dataset_path=dataset_Inflammed_path)   #%%

    if prediction_on_image:
        logger.info("Start prediction on single image...")
        pred_on_single_image(model_folder=model_folder_inf,single_image_path=single_image_path)   #%%

    if testonWSI:
        logger.info("Start prediction on WSI...")
        TestOnSingleSlide(model_folder=model_folder_inf, slidepath=test_slidepath)   #%%

    if testonAllWSIS:
        # Start the timer
        start_time = time.time()
        logger.info("Start prediction on WSI...")
        TestOnSlides(model_folder_inf=model_folder_inf, model_folder_tissue=model_folder_tissue)
        end_time = time.time()
        duration = end_time - start_time
        print(f"All slides have been testet in {duration} seconds")

    if printLossCurves:
        logger.info("Start printing loss/accuracy curves...")
        plot_loss_acc_curves(model_folder=model_folder_inf)  #%%

    if train_vit:
        logger.info("Start training Visual Transformer...")
        model_folder = trainVIT(dataset_path=dataset_Tissue_path)
        logger.info("Congratulations, training your vision transformer was successful!" + str(model_folder))

    if train_2_class_model:
        model_folder = train_new_3model(dataset_path=dataset_Inf_tom_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if train_inf_model:
        model_folder = train_new_inf_model(dataset_path=dataset_Inf_tom_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if train_effNet:
        model_folder = train_new_EFFICIENTNET(dataset_path=dataset_Tissue_path, tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if test_3model:
        logger.info("Start testing the model..")
        print_3model_metrices(model_folder='models/TransferLearning_model_17112023_1755', test_folder=test_Tom_Inf_folder)   #%%

    if evaluate_experiment_tile_level:
        logger.info("Start evaluating the results of the experiment to evaluate human performance")
        evaluate_Experiment_Tiles(csvpathInflammation= "results experiment/Experiment Human Performance Inflamed.csv",
                                  csvpathTissue="results experiment/Experiment Human Performance Tissue.csv")
#%%
