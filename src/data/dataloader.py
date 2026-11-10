"""
PulmoVision — 2.5D Dataset and DataLoader Pipeline
Loads multi-slice CT triplets, processes them through CTPreprocessor,
and delivers stacked [B, 3, 224, 224] batches to deep learning models.
"""

from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import CTPreprocessor


class CT25DDataset(Dataset):
    """
    PyTorch Dataset for 2D and 2.5D CT multi-slice classification.
    
    Args:
        triplets: List of (path_prev, path_curr, path_next, label_str) tuples.
        preprocessor: CTPreprocessor instance handling windowing and augmentations.
        in_channels: Number of channels. 1 for 2D (target slice only), 3 for 2.5D.
    """
    LABEL_MAP = {
        "normal": 0,
        "negative": 0,
        "covid": 1,
        "positive": 1,
    }

    def __init__(
        self,
        triplets: List[Tuple[str, str, str, str]],
        preprocessor: Optional[CTPreprocessor] = None,
        in_channels: int = 3,
    ):
        self.triplets = triplets
        self.preprocessor = preprocessor or CTPreprocessor()
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads slice(s), processes them into [C, 224, 224],
        where C=1 (2D) or C=3 (2.5D).
        
        Returns:
            tensor: [C, 224, 224] float32 tensor
            label: integer 0 (Normal) or 1 (COVID)
        """
        path_prev, path_curr, path_next, label_str = self.triplets[idx]

        if self.in_channels == 1:
            stacked = self.preprocessor(path_curr)  # [1, 224, 224]
        else:
            t_prev = self.preprocessor(path_prev)   # [1, 224, 224]
            t_curr = self.preprocessor(path_curr)   # [1, 224, 224]
            t_next = self.preprocessor(path_next)   # [1, 224, 224]
            stacked = torch.cat([t_prev, t_curr, t_next], dim=0)  # [3, 224, 224]

        # Map string label to class index
        lbl_key = label_str.lower()
        label_idx = self.LABEL_MAP.get(lbl_key, 0)

        return stacked, label_idx


def create_dataloader(
    triplets: List[Tuple[str, str, str, str]],
    preprocessor: Optional[CTPreprocessor] = None,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    in_channels: int = 3,
) -> DataLoader:
    """
    Factory function to construct a PyTorch DataLoader for 2D or 2.5D CT batches.
    """
    dataset = CT25DDataset(triplets=triplets, preprocessor=preprocessor, in_channels=in_channels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader

