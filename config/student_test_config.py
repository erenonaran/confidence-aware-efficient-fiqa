from dataclasses import dataclass

@dataclass
class TestConfig_EdgeNext:
    # Data related
    gfiqa_path_list: list = None
    gfiqa_csv_path = "/content/data/test/GFIQA-20k/mos_val_rating.csv"
    gfiqa_prediction_path = None

    calibration_ratio: float = 0.2
    fitted_curve_path: str = None
    fitted_curve_path_to_save: str = None

    # Training related
    num_epochs: int = 30
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42

    # Optimizer related
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    decay_ratio: float = 0.1
    decay_epochs: int = 10
    loss_type: str = 'mse+plcc'

    # Model related
    model_name: str = 'FIQA_EdgeNeXt_XXS'
    pretrained_path: str  = '/content/best_model.pth'
    model_save_dir: str = None

    # Image processing
    image_size: int = 448
    image_crop: int = 448

    # Device
    gpu_ids: str = "0"

    # Logging
    log_interval: int = 200  # Print log every N batches
    save_interval: int = 1  # Save model every N epochs