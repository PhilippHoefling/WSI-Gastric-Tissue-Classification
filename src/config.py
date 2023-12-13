config_hyperparameter = {
    "seed": 42,
    "dropout": [None],
    "patience": 15,
    "epochs":60,
    "num_workers": 6,
    "class_names": ['0_noninflamed','1_inflamed'],
    #"class_names": ['antrum','corpus'],
    'lr': [1e-2, 1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64, 128],
    #'module__dropout': [0, 0.3, 0.4, 0.5],
    #'max_epochs': [10, 20, 30, 40],
}


#%%

