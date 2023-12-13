
from loguru import logger

from process import load_sort_data
from Tile_inference import tile_inference_regression, tile_inference_binary, pred_on_single_image
from process import plot_file_distribution , get_WSI_aggregation_data
from WSI_Inference import WSI_Test_Pipeline, TestOnSlides, TestOnWSISlideFolder, TestOnWSISlideFolderTissue
from Binary_ResNet18 import train_new_binary_model
from auxiliaries import plot_loss_acc_curves

from evaluaton_human_performance import evaluate_Experiment_Tiles, AI_Tile_prediction_for_Experiment
import time

if __name__ == "__main__":
    # Define dataset paths
    dataset_Tissue_path = 'data/TissueTiles512'
    dataset_final_Tissue = 'data/TissueTilesReworked'
    dataset_Inflammed_path = 'data/InflamedTiles'
    dataset_Inf_tom_path = 'data/InflamedTilesReworked'

    model_folder_inf = "models/TransferLearning_model_05122023_0418"
    model_folder_tissue = "models/TransferLearning_model_09122023_1539"

    tf_model ="imagenet"
    single_image_path = 'data/InflamedTiles/test/inflamed/7CHE d-5_x-58735_y-137940_w-2560_h-2560_inflamed.png'
    test_slidepath = 'E:/Scans/Non-Inflamed/56HE.mrxs'



    #Preprocessing Data
    preprocess = False
    plot_data_distribution = False
    get_WSI_train_Data = False

    # train a new model
    train_gastric_model = False
    train_inf_model = False

    #Test existing models
    test_Tissue_Model = False
    test_Inf_Model = True
    testonWSI =  False
    testonAllWSIS = False
    testonWSIFolder = False
    prediction_on_image = False
    printLossCurves = False

    #Evaluation of Experiment results
    evaluate_experiment_tile_level = False
    evaluate_model_on_experiment_tile_level = False


    if preprocess:
        logger.info("Start preprocessing data...")
        load_sort_data("D:/TomGastric/tiles", dataset_final_Tissue, use_split=True)
        logger.info("Congratulations, the preprocessing was successful!")



    if test_Tissue_Model:
        logger.info("Start testing the model..")
        tile_inference_binary(model_folder=model_folder_tissue, test_folder=dataset_final_Tissue + "/test", safe_wrong_preds=False)

    if test_Inf_Model:
        logger.info("Start testing the model..")
        tile_inference_binary(model_folder=model_folder_inf, test_folder=dataset_Inf_tom_path + "/test" , safe_wrong_preds=True)    #%%

    if plot_data_distribution:
        logger.info("Start analyzing dataset..")
        plot_file_distribution(dataset_path="D:/GastricPlot")   #%%

    if prediction_on_image:
        logger.info("Start prediction on single image...")
        pred_on_single_image(model_folder=model_folder_inf,single_image_path=single_image_path)   #%%

    if testonWSI:
        logger.info("Start prediction on WSI...")
        WSI_Test_Pipeline(model_folder=model_folder_inf, slidepath=test_slidepath)   #%%

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
        plot_loss_acc_curves(model_folder=model_folder_tissue)  #%%


    if train_gastric_model:
        logger.info("Start training a new binary Gastric Model...")
        model_folder = train_new_binary_model(dataset_path=dataset_final_Tissue, tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if train_inf_model:
        logger.info("Start training a new binary Inflammed Model...")
        model_folder = train_new_binary_model(dataset_path=dataset_Inf_tom_path,tf_model=tf_model)
        logger.info("Congratulations, training the baseline models was successful!" + str(model_folder))

    if evaluate_experiment_tile_level:
        logger.info("Start evaluating the results of the experiment to evaluate human performance")
        evaluate_Experiment_Tiles(csvpathInflammation= "results experiment/Experiment Human Performance Inflamed.csvAI.csv",
                                  csvpathTissue="results experiment/Experiment Human Performance Tissue.csvAI.csv")

    if get_WSI_train_Data:
        logger.info("Start analyzing tile data set for WSI aggregation network")
        get_WSI_aggregation_data("D:/TomGastric/tiles")
#%%
    if evaluate_model_on_experiment_tile_level:
        logger.info("Start evaluating the AI model onthe results of the experiment")
        AI_Tile_prediction_for_Experiment(csv_file="results experiment/Experiment Human Performance Tissue.csv",
                                          folder_tiles="C:/Users/phili/OneDrive - Otto-Friedrich-Universität Bamberg/Termin Nürnberg/Tiles_Experiment_Path",
                                          model_folder=model_folder_tissue)
    if testonWSIFolder:
        logger.info("Start prediction on WSI...")
        #TestOnWSISlideFolder(model_folder_inf=model_folder_inf, model_folder_tissue=model_folder_tissue, testfolder='xd' )
        TestOnWSISlideFolderTissue(model_folder_inf=model_folder_inf, model_folder_tissue=model_folder_tissue )
