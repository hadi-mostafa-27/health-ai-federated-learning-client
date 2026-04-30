NOTEBOOK_PROFILES = {
    "kaggle_chexpert_binary": {
        "description": "From notebook63f2593a12 (1).ipynb",
        "backbone": "densenet121",
        "mode": "binary",
        "img_size": 320,
        "class_names": ["negative", "target_label"],
        "normalize_mean": [0.485, 0.485, 0.485],
        "normalize_std": [0.229, 0.229, 0.229],
        "supports_masking": True,
        "target_checkpoint_names": ["fedavg_best.pt", "fedprox_best.pt", "centralized_best.pt"],
    },
    "colab_transfer_multiclass": {
        "description": "From TransferLearning_FL_Colab_Notebook.ipynb",
        "backbone": "resnet18",
        "mode": "multiclass",
        "img_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "supports_masking": False,
        "target_checkpoint_names": ["model.pt"],
    },
}
