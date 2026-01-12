import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from pathlib import Path

class FIQADataset(Dataset):
    def __init__(self, path_list, csv_path, transform=None, is_train=True, train_ratio=0.8, seed=42):

        self.df = pd.read_csv(csv_path, header=None, names=['filename', 'score'])

        train_indices, test_indices = train_test_split(
            np.arange(len(self.df)),
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )

        self.indices = train_indices if is_train else test_indices

        self.path_list = path_list
        self.transform = transform
        self.is_train = is_train
        self.filename_path_mapping = {
            Path(path).name: str(path)
            for path in path_list
        }


    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        row = self.df.iloc[real_idx]
        image_name = str(row['filename'])
        # Add .png extension if filename doesn't have an extension
        if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_name += '.png'
        image_path = self.filename_path_mapping[image_name]

        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            label = torch.tensor(float(row['score']), dtype=torch.float32)

            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return self.__getitem__(0) if idx != 0 else None

    def __len__(self):
        return len(self.indices)



class FIQADatasetWithWebFace(Dataset):
    def __init__(self,
                 path_list,
                 webface_path_list,
                 csv_path,
                 webface_csv_path,
                 webface_confidence_csv_path,
                 is_train=True,
                 transform=None,
                 train_ratio=0.8,
                 seed=42):
        self.df_original = pd.read_csv(csv_path, header=None, names=['filename', 'score'])
        self.df_original['source'] = 'original'
        self.df_original['confidence'] = 1

        self.df_webface = pd.read_csv(webface_csv_path)
        self.df_webface['source'] = 'webface'

        self.df_webface_confidence = pd.read_csv(webface_confidence_csv_path)[['filename','confidence']]
        self.df_webface = pd.merge(self.df_webface, self.df_webface_confidence, on='filename', how='left')


        train_indices, val_indices = train_test_split(
            np.arange(len(self.df_original)),
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )

        self.train_df = pd.concat([
            self.df_original.iloc[train_indices],
            self.df_webface
        ], ignore_index=True)

        self.val_df = self.df_original.iloc[val_indices]

        if is_train:
            self.df = self.train_df
        else:
            self.df = self.val_df

        self.df['confidence'] = self.df['confidence'].fillna(1)


        self.path_list = path_list
        self.webface_path_list = webface_path_list
        self.transform = transform
        self.is_train = is_train

        self.filename_path_mapping = {
            Path(path).name: str(path)
            for path in path_list
        }

        self.webface_filename_path_mapping = {
            Path(path).name: str(path)
            for path in webface_path_list
        }


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_name = str(row['filename'])

            if not any(image_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                image_name += '.png'

            data_source = row['source']
            if data_source == 'original':
              image_path = self.filename_path_mapping[image_name]
            elif data_source == 'webface':
              image_path = self.webface_filename_path_mapping[image_name]

            image = Image.open(image_path).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)
            label = torch.tensor(float(row['score']), dtype=torch.float32)
            confidence = torch.tensor(float(row['confidence']), dtype=torch.float32)

            return image, label, confidence
        except Exception as e:
            print(f"Error loading image {image_name} from {data_source} dataset: {str(e)}")
            # If error occurs, return first image in dataset (rarely happens)
            return self.__getitem__(0) if idx != 0 else None