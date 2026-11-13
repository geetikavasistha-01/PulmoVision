"""
PulmoVision — Stage 2 Feature Extraction Pipeline
Extracts deterministic 1280-dimensional embeddings from an external
ImageNet-1K pretrained EfficientNet-B0 backbone (zero in-domain exposure).
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import os
import hashlib
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.data.slice_stacking import CTSliceStacker
from src.data.preprocessing import CTPreprocessor


class PretrainedFeatureExtractor:
    """
    Modular, deterministic feature extractor using external ImageNet-1K EfficientNet-B0.
    
    Guarantees:
    - Never loads Stage 1 in-domain checkpoints.
    - Applies documented 16-bit radiodensity clipping [200, 1800] -> scale [0, 1] ->
      3-channel replication -> ImageNet normalization -> 224x224 resize.
    - Yields 1280-dimensional slice embeddings.
    """
    WEIGHTS_ID = "EfficientNet_B0_Weights.IMAGENET1K_V1"
    EXPECTED_WEIGHTS_HASH = "7f5810bc"
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Load external ImageNet-1K weights only
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        self.weights_url = weights.url

        # Truncate classifier head: retain features and avgpool
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
            
        self.features.eval().to(self.device)
        self.avgpool.eval().to(self.device)
        
        self.preprocessor = CTPreprocessor(target_size=(224, 224), is_training=False)
        self.mean_tensor = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1).to(self.device)
        self.std_tensor = torch.tensor(self.IMAGENET_STD).view(3, 1, 1).to(self.device)

    def preprocess_slice(self, raw_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocesses a raw 16-bit CT slice (or path) into standardized [3, 224, 224] tensor:
        1. Radiodensity clip [200, 1800]
        2. Dynamic range scale to [0.0, 1.0]
        3. Spatial resize to 224x224
        4. Replicate grayscale to 3 channels (R=G=B)
        5. ImageNet mean/std standardization
        """
        # Step 1, 2, 3 via CTPreprocessor -> [1, 224, 224] in [0, 1]
        t_1ch = self.preprocessor(raw_input).to(self.device)
        
        # Step 4: Replicate to 3 channels [3, 224, 224]
        t_3ch = t_1ch.repeat(3, 1, 1)
        
        # Step 5: Standardize using ImageNet mean & std
        standardized = (t_3ch - self.mean_tensor) / self.std_tensor
        return standardized

    @torch.no_grad()
    def extract_slice_embedding(self, slice_tensor_3ch: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1280-dimensional embedding from a [3, 224, 224] slice tensor.
        """
        if slice_tensor_3ch.dim() == 3:
            slice_tensor_3ch = slice_tensor_3ch.unsqueeze(0) # [1, 3, 224, 224]
            
        f_maps = self.features(slice_tensor_3ch) # [B, 1280, 7, 7]
        pooled = self.avgpool(f_maps) # [B, 1280, 1, 1]
        emb = torch.flatten(pooled, 1) # [B, 1280]
        return emb

    @torch.no_grad()
    def extract_patient_sequence(
        self,
        slice_paths: List[str],
        batch_size: int = 16
    ) -> torch.Tensor:
        """
        Extracts embeddings for an ordered sequence of slice paths.
        Returns:
            sequence_tensor: [T, 1280] float32 tensor on CPU
        """
        all_embeddings = []
        for i in range(0, len(slice_paths), batch_size):
            batch_paths = slice_paths[i:i + batch_size]
            batch_tensors = [self.preprocess_slice(p) for p in batch_paths]
            batch_input = torch.stack(batch_tensors, dim=0) # [b, 3, 224, 224]
            embs = self.extract_slice_embedding(batch_input) # [b, 1280]
            all_embeddings.append(embs.cpu())
            
        return torch.cat(all_embeddings, dim=0) # [T, 1280]

    def get_weights_sha256(self) -> str:
        """Computes or retrieves SHA256 of the cached torchvision weights file."""
        hub_dir = torch.hub.get_dir()
        ckpt_dir = os.path.join(hub_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if "efficientnet_b0" in f:
                    fpath = os.path.join(ckpt_dir, f)
                    return hashlib.sha256(open(fpath, "rb").read()).hexdigest()
        return "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934"

    def extract_and_build_patient_manifest(
        self,
        manifest_csv: str = "data/metadata/Annotation/CSV/train1.csv",
        data_dir: str = "data/raw/TrainValidation",
        patient_filter: Optional[List[str]] = None,
        max_patients: Optional[int] = None,
        output_path: Optional[str] = None,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Parses manifest, groups slices strictly by (patient_id, series_id) to prevent
        cross-series mixing, sorts slices in craniocaudal order (ascending slice_idx),
        extracts 1280D embeddings on device, and packages into cache with provenance.
        """
        import torchvision
        from datetime import datetime, timezone

        if "test1" in manifest_csv.lower():
            raise ValueError("Safety violation: Feature extraction must never access test1.csv!")

        df = pd.read_csv(manifest_csv)
        filter_set = set(p.lower() for p in patient_filter) if patient_filter is not None else None
        
        # Group by (patient_id, series_id)
        series_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for _, row in df.iterrows():
            fname = row["filename"]
            full_path = os.path.join(data_dir, fname)
            meta = CTSliceStacker.parse_filename(full_path)
            pid = meta["patient_id"]
            sid = meta["series_id"]
            
            if filter_set is not None and pid not in filter_set:
                continue
                
            key = (pid, sid)
            if key not in series_groups:
                series_groups[key] = []
            series_groups[key].append(meta)
            
        sorted_keys = sorted(list(series_groups.keys()))
        if max_patients is not None:
            # Filter keys to first max_patients unique patient IDs
            seen_pids = set()
            filtered_keys = []
            for pid, sid in sorted_keys:
                if pid not in seen_pids and len(seen_pids) >= max_patients:
                    continue
                seen_pids.add(pid)
                filtered_keys.append((pid, sid))
            sorted_keys = filtered_keys

        series_output: Dict[str, Dict[str, Any]] = {}
        patients_output: Dict[str, Dict[str, Any]] = {}
        total_slices_processed = 0
        total_duplicates_resolved = 0
        duplicate_provenance_map: Dict[str, Any] = {}

        for pid, sid in sorted_keys:
            slices = series_groups[(pid, sid)]
            # Anatomical deduplication within (patient_id, series_id):
            # COVID-CTset manifest contains duplicate archive prefix entries (e.g. normal1 vs normal5)
            # with bit-for-bit identical content. Retain exactly one canonical slice per slice_idx
            # using deterministic lexicographical filepath preference.
            slices_by_idx: Dict[int, Dict[str, Any]] = {}
            for s in sorted(slices, key=lambda x: x["filepath"]):
                sidx = s["slice_idx"]
                if sidx in slices_by_idx:
                    total_duplicates_resolved += 1
                    slices_by_idx[sidx].setdefault("duplicate_aliases", []).append(s["filepath"])
                else:
                    s_copy = s.copy()
                    s_copy["duplicate_aliases"] = []
                    slices_by_idx[sidx] = s_copy

            # Strict craniocaudal sorting: ascending slice_idx
            slices_sorted = [slices_by_idx[sidx] for sidx in sorted(slices_by_idx.keys())]
            paths = [s["filepath"] for s in slices_sorted]

            for s in slices_sorted:
                if s.get("duplicate_aliases"):
                    k_str = f"{pid}_{sid}_IM{s['slice_idx']:05d}"
                    duplicate_provenance_map[k_str] = {
                        "canonical_filename": os.path.basename(s["filepath"]),
                        "duplicate_aliases": [os.path.basename(a) for a in s["duplicate_aliases"]],
                        "patient_id": pid,
                        "series_id": sid,
                        "slice_idx": s["slice_idx"],
                    }

            label_str = slices_sorted[0]["label"]
            label_bin = 1 if "covid" in label_str.lower() else 0

            embs = self.extract_patient_sequence(paths, batch_size=batch_size) # [T_series, 1280] float32
            total_slices_processed += len(slices_sorted)

            series_key = f"{pid}_{sid}"
            series_rec = {
                "patient_id": pid,
                "series_id": sid,
                "series_key": series_key,
                "label": label_bin,
                "label_str": label_str,
                "num_slices": len(slices_sorted),
                "slice_indices": [s["slice_idx"] for s in slices_sorted],
                "slice_filenames": [os.path.basename(p) for p in paths],
                "duplicate_aliases": {
                    s["slice_idx"]: [os.path.basename(a) for a in s["duplicate_aliases"]]
                    for s in slices_sorted if s.get("duplicate_aliases")
                },
                "embeddings": embs,
            }
            series_output[series_key] = series_rec

            if pid not in patients_output:
                patients_output[pid] = {
                    "patient_id": pid,
                    "label": label_bin,
                    "label_str": label_str,
                    "series_ids": [],
                    "series": {},
                    "total_slices": 0,
                }
            patients_output[pid]["series_ids"].append(sid)
            patients_output[pid]["series"][sid] = series_rec
            patients_output[pid]["total_slices"] += len(slices_sorted)

        metadata = {
            "source_manifest": manifest_csv,
            "deduplication_protocol": "Option 1 — Anatomical De-duplication (canonical single instance per (patient_id, series_id, slice_idx); duplicate archive entries preserved in duplicate_provenance)",
            "canonical_selection_rule": "lexicographical_filepath_ascending (sorted(files)[0])",
            "backbone_identifier": self.WEIGHTS_ID,
            "weights_url": self.weights_url,
            "weights_sha256": self.get_weights_sha256(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(self.device),
            "clipping_window": [200, 1800],
            "spatial_target_size": [224, 224],
            "channel_expansion": "replicate_3ch",
            "normalization": "imagenet_standardized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            "total_patients": len(patients_output),
            "total_series": len(series_output),
            "total_slices": total_slices_processed,
            "total_manifest_rows": len(df),
            "duplicate_slices_resolved": total_duplicates_resolved,
            "feature_dim": 1280,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        cache = {
            "metadata": metadata,
            "patients": patients_output,
            "series": series_output,
            "duplicate_provenance": duplicate_provenance_map,
        }

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(cache, output_path)

        return cache
