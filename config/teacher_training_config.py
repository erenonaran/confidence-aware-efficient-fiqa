from dataclasses import dataclass

@dataclass
class TrainingConfig_SwinB:
    # Data related
    data_dir: str = '/content/data/train'
    csv_path: str = '/content/train.csv'
    webface_path_list: list = None
    webface_csv_path = '/content/webface_pseudolabels_augmented.csv'

    train_ratio: float = 0.8
    fitted_curve_path: str = '/content/project/efficient_fiqa/checkpoints/logistic_curve.pt'
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
    model_name: str = 'FIQA_Swin_B'
    pretrained_path: str  = '/content/project/efficient_fiqa/checkpoints/Swin_B_plus_checkpoint.pt'
    model_save_dir: str = None

    # Image processing
    image_size: int = 448
    image_crop: int = 448

    # Device
    gpu_ids: str = "0"

    # Logging
    log_interval: int = 200  # Print log every N batches
    save_interval: int = 1  # Save model every N epochs