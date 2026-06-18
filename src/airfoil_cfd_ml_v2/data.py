from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class TurbDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        if not self.file_list:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        file_path = os.path.join(self.data_dir, self.file_list[index])
        npz_data = np.load(file_path)
        data_array = npz_data["a"]  # shape (6, 128, 128)

        input_data = data_array[:3, :, :].astype(np.float32)
        target_data = data_array[3:, :, :].astype(np.float32)

        inputs_tensor = torch.from_numpy(input_data)
        targets_tensor = torch.from_numpy(target_data)
        return inputs_tensor, targets_tensor


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: TurbDataset


def create_splits_and_loaders(
    train_dir: str,
    test_dir: str,
    batch_size: int,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    shuffle_train: bool = True,
) -> LoaderBundle:
    full_train = TurbDataset(train_dir)
    test_dataset = TurbDataset(test_dir)

    dataset_size = len(full_train)
    train_size = int(dataset_size * (1 - val_ratio))
    val_size = dataset_size - train_size

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
