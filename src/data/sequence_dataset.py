"""
PulmoVision — Patient-Level CT Sequence Dataset
Constructs craniocaudally sorted axial slice sequences per patient,
supporting variable sequence lengths, edge cases (T=1), zero-padding,
and boolean attention masks.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.data.slice_stacking import CTSliceStacker
from src.data.preprocessing import CTPreprocessor


class CTPatientSequenceDataset(Dataset):
    """
    Dataset that loads an entire patient CT volume as an ordered sequence of 2D slices.
    """
    def __init__(
        self,
        manifest_csv: str = "data/metadata/Annotation/CSV/train1.csv",
        data_dir: str = "data/raw/TrainValidation",
        patient_ids: Optional[List[str]] = None,
        preprocessor: Optional[CTPreprocessor] = None,
    ):
        self.data_dir = data_dir
        self.preprocessor = preprocessor or CTPreprocessor(target_size=(224, 224), is_training=False)
        
        df = pd.read_csv(manifest_csv)
        self.patients_data = self._build_patient_sequences(df, patient_ids)
        self.patient_list = list(self.patients_data.keys())

    def _build_patient_sequences(
        self,
        df: pd.DataFrame,
        patient_filter: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        filter_set = set(p.lower() for p in patient_filter) if patient_filter is not None else None
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        
        for _, row in df.iterrows():
            fname = row["filename"]
            full_path = os.path.join(self.data_dir, fname)
            meta = CTSliceStacker.parse_filename(full_path)
            pid = meta["patient_id"]
            
            if filter_set is not None and pid not in filter_set:
                continue
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(meta)
            
        patients: Dict[str, Dict[str, Any]] = {}
        for pid, slices in grouped.items():
            slices_sorted = sorted(slices, key=lambda s: s["slice_idx"])
            label_str = slices_sorted[0]["label"]
            label_bin = 1 if "covid" in label_str.lower() else 0
            
            patients[pid] = {
                "patient_id": pid,
                "label": label_bin,
                "label_str": label_str,
                "slice_paths": [s["filepath"] for s in slices_sorted],
                "slice_indices": [s["slice_idx"] for s in slices_sorted],
                "num_slices": len(slices_sorted),
            }
        return patients

    def __len__(self) -> int:
        return len(self.patient_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.patient_list[idx]
        p_info = self.patients_data[pid]
        
        tensors = []
        for path in p_info["slice_paths"]:
            img = self.preprocessor.load_16bit_tiff(path)
            processed = self.preprocessor.preprocess(img)
            t_3ch = torch.from_numpy(processed).unsqueeze(0).repeat(3, 1, 1).float()
            tensors.append(t_3ch)
            
        sequence_tensor = torch.stack(tensors, dim=0)
        return {
            "slices": sequence_tensor,
            "label": p_info["label"],
            "patient_id": pid,
            "num_slices": p_info["num_slices"],
            "slice_indices": p_info["slice_indices"],
        }


class CachedEmbeddingDataset(Dataset):
    """
    Fast sequence dataset operating on pre-extracted 1280D feature vectors.
    Supports both series-level sequences and patient-level dictionary entries.
    Implements equal patient weighting so multiple series from one patient
    do not inflate that patient's contribution to training loss.
    """
    def __init__(
        self,
        features_dict: Dict[str, Dict[str, Any]],
        patient_ids: Optional[List[str]] = None,
    ):
        if "series" in features_dict:
            features_dict = features_dict["series"]

        if patient_ids is not None:
            pid_set = set(p.lower() for p in patient_ids)
            filtered = {}
            for k, v in features_dict.items():
                p_id = v.get("patient_id", k)
                if p_id.lower() in pid_set:
                    filtered[k] = v
            self.data = filtered
        else:
            self.data = features_dict
            
        self.item_keys = sorted(list(self.data.keys()))

        # Compute series frequency per patient to enforce equal patient weighting:
        # Each series is assigned weight w = 1.0 / N_series(patient),
        # ensuring the sum of series weights for any patient is strictly 1.0.
        patient_series_counts: Dict[str, int] = {}
        for k, v in self.data.items():
            pid = v.get("patient_id", k).lower()
            patient_series_counts[pid] = patient_series_counts.get(pid, 0) + 1
        self.patient_series_counts = patient_series_counts

    def __len__(self) -> int:
        return len(self.item_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.item_keys[idx]
        item = self.data[key]
        pid = item.get("patient_id", key).lower()
        n_series = self.patient_series_counts.get(pid, 1)
        patient_weight = 1.0 / n_series

        return {
            "embeddings": item["embeddings"], # [T, 1280]
            "label": item["label"],
            "patient_id": item.get("patient_id", key),
            "series_id": item.get("series_id", "default"),
            "num_slices": item["num_slices"],
            "slice_indices": item.get("slice_indices", []),
            "patient_weight": float(patient_weight),
            "num_patient_series": int(n_series),
        }


class PatientSequenceDataset(Dataset):
    """
    Patient-level dataset that groups all scan series belonging to each patient.
    Guarantees strict patient-level isolation and enables exact equal patient weighting.
    """
    def __init__(
        self,
        features_dict: Dict[str, Any],
        patient_ids: Optional[List[str]] = None,
    ):
        if "patients" in features_dict:
            raw_patients = features_dict["patients"]
        else:
            raw_patients = features_dict

        if patient_ids is not None:
            pid_set = set(p.lower() for p in patient_ids)
            self.patients = {pid: v for pid, v in raw_patients.items() if pid.lower() in pid_set}
        else:
            self.patients = raw_patients

        self.patient_list = sorted(list(self.patients.keys()))

    def __len__(self) -> int:
        return len(self.patient_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.patient_list[idx]
        p_data = self.patients[pid]

        series_items = []
        for sid, s_rec in p_data["series"].items():
            series_items.append({
                "patient_id": pid,
                "series_id": sid,
                "label": s_rec["label"],
                "num_slices": s_rec["num_slices"],
                "slice_indices": s_rec["slice_indices"],
                "slice_filenames": s_rec.get("slice_filenames", []),
                "embeddings": s_rec["embeddings"], # [T_s, 1280] float32
            })

        return {
            "patient_id": pid,
            "label": p_data["label"],
            "num_series": len(series_items),
            "series": series_items,
        }


def collate_sequence_embeddings(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching variable-length sequence embeddings.
    Zero-pads embeddings to T_max and constructs boolean attention mask.
    """
    embeddings = [item["embeddings"] for item in batch]
    lengths = torch.tensor([item["num_slices"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    patient_ids = [item["patient_id"] for item in batch]
    patient_weights = torch.tensor(
        [item.get("patient_weight", 1.0) for item in batch], dtype=torch.float32
    )

    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    b_size, t_max, _ = padded_embeddings.shape
    mask = torch.zeros(b_size, t_max, dtype=torch.float32)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0

    return {
        "embeddings": padded_embeddings,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
        "patient_ids": patient_ids,
        "patient_weights": patient_weights,
    }


def collate_patient_balanced_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for patient-balanced batching.
    Flattens all series across B patients into padded sequence tensors,
    and returns patient index mapping to enable exact patient-level loss averaging.
    """
    all_embeddings = []
    all_lengths = []
    series_patient_indices = []
    patient_labels = []
    patient_ids = []
    num_series_per_patient = []

    for p_idx, p_item in enumerate(batch):
        pid = p_item["patient_id"]
        lbl = p_item["label"]
        patient_ids.append(pid)
        patient_labels.append(lbl)
        num_series_per_patient.append(p_item["num_series"])

        for s in p_item["series"]:
            all_embeddings.append(s["embeddings"])
            all_lengths.append(s["num_slices"])
            series_patient_indices.append(p_idx)

    padded_embeddings = pad_sequence(all_embeddings, batch_first=True, padding_value=0.0)
    M, t_max, _ = padded_embeddings.shape
    mask = torch.zeros(M, t_max, dtype=torch.float32)
    for i, length in enumerate(all_lengths):
        mask[i, :length] = 1.0

    return {
        "embeddings": padded_embeddings,                            # [M, T_max, 1280]
        "mask": mask,                                              # [M, T_max]
        "lengths": torch.tensor(all_lengths, dtype=torch.long),    # [M]
        "series_patient_indices": torch.tensor(series_patient_indices, dtype=torch.long), # [M]
        "patient_labels": torch.tensor(patient_labels, dtype=torch.long),                 # [B]
        "num_series_per_patient": torch.tensor(num_series_per_patient, dtype=torch.long), # [B]
        "patient_ids": patient_ids,                                # list of B patient strings
    }

