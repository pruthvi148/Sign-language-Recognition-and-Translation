import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, max_frames=30, min_samples=5, augment=True):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.augment = augment
        self.data_paths = []
        self.labels = []
        
        # Build vocabulary - ONLY classes with enough samples
        all_classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.classes = []
        for cls_name in all_classes:
            cls_dir = self.data_dir / cls_name
            num_files = len(list(cls_dir.glob('*.npy')))
            if num_files >= min_samples:
                self.classes.append(cls_name)
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            for npy_file in cls_dir.glob('*.npy'):
                self.data_paths.append(npy_file)
                self.labels.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.data_paths)
        
    def __getitem__(self, idx):
        path = self.data_paths[idx]
        keypoints = np.load(str(path))
        
        # Data augmentation
        if self.augment and np.random.rand() > 0.3:
            # Spatial jitter
            noise = np.random.normal(0, 0.005, keypoints.shape)
            keypoints = keypoints + noise
            # Random scale
            scale = np.random.uniform(0.9, 1.1)
            keypoints = keypoints * scale
            # Random time masking (zero out a few frames)
            if keypoints.shape[0] > 5:
                mask_len = np.random.randint(1, 4)
                mask_start = np.random.randint(0, keypoints.shape[0] - mask_len)
                keypoints[mask_start:mask_start+mask_len] = 0

        # Ensure max_frames shape
        if keypoints.shape[0] < self.max_frames:
            padding = self.max_frames - keypoints.shape[0]
            keypoints = np.pad(keypoints, ((0, padding), (0, 0)), mode='constant')
        elif keypoints.shape[0] > self.max_frames:
            max_start = keypoints.shape[0] - self.max_frames
            start_idx = np.random.randint(0, max_start + 1)
            keypoints = keypoints[start_idx : start_idx + self.max_frames]
            
        # Velocity Feature Extraction
        deltas = np.zeros_like(keypoints)
        deltas[1:] = keypoints[1:] - keypoints[:-1]
        features = np.concatenate([keypoints, deltas], axis=-1)
            
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
