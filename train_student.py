import os
import time
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from config import student_training_config
from data import FIQA_dataset
from model import Efficient_FIQA_models
from augmentation import make_aug_transforms
from calibration import fit_logistic_mapping, performance_fit, logistic_func

def optimizer_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def kd_feat_loss(fs, ft):
    fs = F.normalize(fs, dim=1)
    ft = F.normalize(ft, dim=1)
    return F.mse_loss(fs, ft)

def setup_logger(save_dir):
    """Set up logging configuration"""
    import logging
    log_file = save_dir / 'train.log'

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_print(message, logger, level="info"):
    print(message)

    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)


def get_transforms(config):
    """Get data augmentation and preprocessing methods"""
    train_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(config.image_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def validate(model, val_loader, criterion1, criterion2, device, epoch, writer, logger):
    """Validate model performance"""
    model.eval()
    val_loss = 0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion1(outputs, labels.view(-1, 1)) + criterion2(outputs, labels.view(-1, 1))

            val_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    predictions = np.array(predictions).squeeze()
    ground_truth = np.array(ground_truth)

    plcc, srcc, krcc, rmse = performance_fit(ground_truth, predictions)

    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/SRCC', srcc, epoch)
    writer.add_scalar('Val/PLCC', plcc, epoch)
    writer.add_scalar('Val/RMSE', rmse, epoch)

    log_print(f"Validation Epoch {epoch}: Loss={val_loss:.4f}, SRCC={srcc:.4f}, "
                f"KRCC={krcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}", logger, "info")

    return val_loss, srcc, plcc



def main():
    print("train png:", len(list(glob('/content/data/train/*.png'))))
    print("webface jpg:", len(list(glob('/content/webface/**/*.jpg'))))

    config = student_training_config.TrainingConfig_EdgeNext(
        path_list= list(glob('/content/data/train/*.png')),
        webface_path_list= list(glob('/content/webface/**/*.jpg'))
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(config.model_save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger(save_dir)
    writer = SummaryWriter(save_dir / 'runs')

    log_print("\nTraining Configuration:", logger, "info")
    log_print("-" * 100, logger, "info")
    for key, value in config.__dict__.items():
        if key == 'path_list' or key == 'webface_path_list':
            continue
        log_print(f"{key}: {value}", logger, "info")
    log_print("-" * 100 + "\n", logger, "info")

    log_print(f"Using GPUs: {config.gpu_ids}", logger, "info")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_transform, val_transform = get_transforms(config)


    train_dataset = FIQA_dataset.FIQADatasetWithWebFace(
        path_list=config.path_list,
        webface_path_list=config.webface_path_list,
        csv_path=config.csv_path,
        webface_csv_path=config.webface_csv_path,
        webface_confidence_csv_path=config.webface_confidence_csv_path,
        transform=train_transform,
        is_train=True,
        train_ratio=config.train_ratio
    )

    val_dataset = FIQA_dataset.FIQADatasetWithWebFace(
        path_list=config.path_list,
        webface_path_list=config.webface_path_list,
        csv_path=config.csv_path,
        webface_csv_path=config.webface_csv_path,
        webface_confidence_csv_path=config.webface_confidence_csv_path,
        transform=val_transform,
        is_train=False,
        train_ratio=config.train_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    log_print(f"Train dataset size: {len(train_dataset)}", logger, "info")
    log_print(f"Val dataset size: {len(val_dataset)}", logger, "info")

    teacher = Efficient_FIQA_models.FIQA_Swin_B(pretrained_path=config.teacher_pretrained_path, is_pretrained=True)

    student = Efficient_FIQA_models.FIQA_EdgeNeXt_XXS(is_pretrained=True)

    proj = torch.nn.Linear(168, 1024)

    param_num = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log_print(f'Trainable params: {param_num/1e6:.2f} million', logger, "info")

    criterion1 = nn.MSELoss()
    criterion2 = plcc_loss

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(proj.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.decay_epochs,
        gamma=config.decay_ratio
    )
    best_score = -1  # Use average of PLCC and SRCC as metric
    best_epoch = 0
    start_epoch = 0
    start_time = time.time()

    scaler = torch.amp.GradScaler()

    if os.path.exists(save_dir / 'last_model.pth'):
      ckpt = torch.load(save_dir / 'last_model.pth', map_location="cpu", weights_only=False)

      student.load_state_dict(ckpt["student_state_dict"])
      proj.load_state_dict(ckpt["proj_state_dict"])
      optimizer.load_state_dict(ckpt["optimizer_state_dict"])
      optimizer_to(optimizer, device)
      scheduler.load_state_dict(ckpt["scheduler_state_dict"])
      scaler.load_state_dict(ckpt["scaler_state_dict"])

      start_epoch = ckpt["epoch"] + 1
      best_score = ckpt.get("best_score", best_score)

      print(f"Resuming from epoch {start_epoch}, best_score={best_score:.4f}")

    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
      p.requires_grad = False

    student = student.to(device)
    proj = proj.to(device)

    for epoch in range(start_epoch, config.num_epochs):
        student.train()
        proj.train()

        train_loss = 0
        batch_start_time = time.time()

        for batch_idx, (images, labels, confidences) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)
            conf   = confidences.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                pred_s, feat_s = student(images, return_feat=True)
                pred_s = pred_s.view(-1, 1)

                with torch.no_grad():
                    _, feat_t = teacher(images, return_feat=True)

                feat_s_proj = proj(feat_s)
                loss_kd = kd_feat_loss(feat_s_proj, feat_t)

                loss_mse  = (conf * (pred_s - labels).pow(2)).mean()
                loss_plcc = plcc_loss(pred_s, labels)

                loss = loss_mse + conf.mean() * loss_plcc + 0.1 * loss_kd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (batch_idx + 1) % config.log_interval == 0:
                avg_loss = train_loss / (batch_idx + 1)
                batch_time = time.time() - batch_start_time
                log_print(
                    f"Epoch [{epoch+1}/{config.num_epochs}], "
                    f"Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Time: {batch_time:.2f}s", logger, "info"
                )
                writer.add_scalar('Train/BatchLoss', avg_loss,
                                epoch * len(train_loader) + batch_idx)
                batch_start_time = time.time()

        train_loss /= len(train_loader)
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)

        val_loss, srcc, plcc = validate(student, val_loader, criterion1, criterion2, device, epoch, writer, logger)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/LearningRate', current_lr, epoch)

        # Calculate current performance score (0.5×PLCC + 0.5×SRCC)
        current_score = (plcc + srcc) / 2

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": student.state_dict(),
                    "proj_state_dict": proj.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_score": best_score,
                    "best_plcc": plcc,
                    "best_srcc": srcc,
                },
                save_dir / "best_model.pth"
            )

            log_print(
                f"New best model saved at epoch {epoch+1} with score {current_score:.4f} "
                f"(PLCC: {plcc:.4f}, SRCC: {srcc:.4f})",
                logger,
                "info"
            )

        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "proj_state_dict": proj.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_score,
                "best_plcc": plcc,
                "best_srcc": srcc,
            },
            save_dir / "last_model.pth"
        )

        log_print(
            f"Saved LAST checkpoint at epoch {epoch+1} (best_score={best_score:.4f})",
            logger,
            "info",
        )


    # Training completed, record total time
    total_time = time.time() - start_time
    log_print(f"Training completed in {total_time/3600:.2f} hours", logger, "info")
    log_print(f"Best Score: {best_score:.4f} (Epoch {best_epoch+1})", logger, "info")

    writer.close()

if __name__ == '__main__':
    main()