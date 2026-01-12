import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob
from scipy.optimize import curve_fit

from config import teacher_training_config
from model import Efficient_FIQA_models
from augmentation import make_aug_transforms
from calibration import fit_logistic_mapping, performance_fit, logistic_func


def setup_device(gpu_ids):
    gpu_ids = str(gpu_ids).strip()
    if gpu_ids.lower() == 'cpu':
        return torch.device('cpu')

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available, using CPU instead")
        return torch.device('cpu')
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = torch.device("cuda")
        print(f"Using GPU {gpu_ids}")
        return device
    except Exception as e:
        print(f"Error setting up GPU: {str(e)}")
        print("Falling back to CPU")
        return torch.device('cpu')
    


def create_test_transform(image_size, image_crop):
    """Create test image transform"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict_quality(model, image_list, transform, device):
    try:
        batch = torch.stack([transform(img) for img in image_list]).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(batch).squeeze().tolist()
        if isinstance(outputs, float):
            outputs = [outputs]
        if len(outputs) != len(image_list):
           raise ValueError(
               f"Output length mismatch: got {len(outputs)} outputs for {len(image_list)} images."
           )

        return outputs
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None



def main():
    webface_path_list = list(glob.glob('/content/webface/**/*.jpg'))
    config = teacher_training_config.TrainingConfig_SwinB(webface_path_list=webface_path_list)

    device = setup_device(config.gpu_ids)

    if config.model_name == 'FIQA_Swin_B':
        model = Efficient_FIQA_models.FIQA_Swin_B(pretrained_path=None)
    print("Using model: ", config.model_name)

    try:
        if device.type == 'cpu':
            state_dict = torch.load(os.path.join(config.pretrained_path), map_location='cpu')
        else:
            state_dict = torch.load(os.path.join(config.pretrained_path))

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully and moved to {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    transform = create_test_transform(config.image_size, config.image_crop)

    aug_transform = make_aug_transforms(
        image_size=config.image_size,
        crop_size=config.image_crop,
    )

    if config.fitted_curve_path is not None and os.path.exists(config.fitted_curve_path):
        print(f"Loading fitted logistic curve from {config.fitted_curve_path}")
        mapping_params = torch.load(config.fitted_curve_path, map_location="cpu").numpy()
    else:
        print("Prediction on original dataset...")

        df_data = pd.read_csv(config.csv_path, header=None, names=['filename', 'score'])

        path_list = []
        path_score_mapping = {}
        for _, row in df_data.iterrows():
            image_name = str(row['filename'])
            if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                image_name += '.png'
            img_path = os.path.join(config.data_dir, image_name)
            path_list.append(img_path)
            path_score_mapping[img_path] = float(row['score'])

        print(f"Images: {len(path_list)}")
        batches = [
              path_list[i:i + config.batch_size]
              for i in range(0, len(path_list), config.batch_size)
        ]

        data_scores = []
        data_gt_scores = []
        for batch in tqdm(batches):
          batch_gt_scores = []
          batch_image_list = []
          for image_path in batch:
              try:
                  image = Image.open(image_path).convert('RGB')
                  batch_image_list.append(image)

                  batch_gt_scores.append(path_score_mapping[image_path])
              except Exception as e:
                  print(f"Error loading {image_path}: {e}")
                  continue

          if len(batch_image_list) == 0:
              print(f"Error: empty batch")
              continue

          batch_scores = predict_quality(model, batch_image_list, transform, device)
          if batch_scores is not None:
            data_scores.extend(batch_scores)
            data_gt_scores.extend(batch_gt_scores)


        print("Fitting logistic mapping...")
        mapping_params = fit_logistic_mapping(np.array(data_gt_scores), np.array(data_scores))
        plcc, srcc, krcc, rmse = performance_fit(np.array(data_gt_scores), np.array(data_scores))
        print(f"SRCC={srcc:.4f}, KRCC={krcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")


        mapping_params = torch.tensor(mapping_params, dtype=torch.float32)
        torch.save(mapping_params, config.fitted_curve_path_to_save)


    if isinstance(config.webface_path_list, str):
       config.webface_path_list = [config.webface_path_list]

    if os.path.exists(config.webface_csv_path):
        scores_df = pd.read_csv(config.webface_csv_path)
        print(f"CSV file loaded from {config.webface_csv_path}")
        existing_files = set(scores_df["filename"])
    else:
        print(f"CSV file created at {config.webface_csv_path}")
        existing_files = set()

    remaining_image_path_list = [
        path for path in config.webface_path_list
        if os.path.basename(str(path)) not in existing_files
    ]
    print(f"Remaining WebFace images: {len(remaining_image_path_list)}")

    webface_batches = [
        remaining_image_path_list[i:i + config.batch_size]
        for i in range(0, len(remaining_image_path_list), config.batch_size)
    ]

    for batch in tqdm(webface_batches):
        batch_filename_list = []
        batch_image_list = []
        for image_path in batch:
            try:
                image = Image.open(image_path).convert('RGB')
                for _ in range(4):
                  batch_image_list.append(image.copy())
                batch_filename_list.append(str(image_path).split('/')[-1])
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

        if len(batch_image_list) == 0:
            print(f"Error: empty batch")
            continue

        batch_scores_raw = predict_quality(model, batch_image_list, aug_transform, device)
        if batch_scores_raw is None:
          continue

        batch_scores_raw = np.array(batch_scores_raw).reshape(len(batch_filename_list), 4)

        batch_scores_aligned = logistic_func(batch_scores_raw, *mapping_params)

        score_mean = batch_scores_aligned.mean(axis=1)
        score_std  = batch_scores_aligned.std(axis=1)

        confidence = np.exp(-score_std)
        if batch_scores_aligned is not None:
            batch_df = pd.DataFrame(
                {"filename": batch_filename_list,
                 "score_mean": score_mean,
                 "score_std": score_std,
                 "confidence": confidence,
            })

            batch_df.to_csv(
                config.webface_csv_path,
                mode="a",
                header=not os.path.exists(config.webface_csv_path),
                index=False
            )


if __name__ == '__main__':
    main()