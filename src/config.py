config_hyperparameter = {
    "seed": 42,
    "dropout": [0.0],
    "lr": [1e-2, 1e-3, 1e-4, 1e-5],
    "batch_size": [16, 32, 64, 128],
    "patience": 15,
    "epochs":50,
    "num_workers": 8,
    "class_names": ['inflamed','noninflamed']
    #"class_names": ['antrum','corpus','intermediate']
    #'lr': [1e-2, 1e-3, 1e-4, 1e-5],
    #'batch_size': [16, 32, 64, 128],
    #'module__dropout': [0, 0.3, 0.4, 0.5],
    #'max_epochs': [10, 20, 30, 40],
}
