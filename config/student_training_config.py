from dataclasses import dataclass
from glob import glob

@dataclass
class TrainingConfig_EdgeNext:
    path_list: list
    webface_path_list: list
    csv_path: str = '/content/train.csv'
    webface_csv_path: str = '/content/webface_pseudolabels_aligned.csv'
    webface_confidence_csv_path: str = '/content/webface_pseudolabels_augmented.csv'
    teacher_pretrained_path: str = '/content/project/efficient_fiqa/checkpoints/Swin_B_plus_checkpoint.pt'
    train_ratio: float = 0.8

    num_epochs: int = 30
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42

    # Optimizer related
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    decay_ratio: float = 0.1
    decay_epochs: int = 10
    loss_type: str = 'mse+plcc'  # Options: 'mse', 'plcc', 'mse+plcc'

    # Model related
    model_save_dir: str = '/content/FIQA_EdgeNeXt/'

    # Image processing
    image_size: int = 352
    image_crop: int = 352

    # Device
    gpu_ids: str = "0"  # GPU IDs to use, separated by commas

    # Logging
    log_interval: int = 200  # Print log every N batches