"""
PulmoVision — 2.5D Slice Stacking Engine
Extracts patient and scan provenance, validates sequential ordering,
and constructs boundary-safe (Z-1, Z, Z+1) slice triplets.
"""

from typing import List, Tuple, Dict
import re
from pathlib import Path


class CTSliceStacker:
    """
    Parses CT filenames, groups slices strictly by (Patient_ID, Series_ID),
    sorts them sequentially, and generates valid 2.5D neighbour triplets.
    """

    @staticmethod
    def parse_filename(filepath: str) -> Dict[str, str]:
        """
        Parses COVID-CTset filenames like:
        '137covid_patient100_SR_2_IM00028.tif' or 'normal_patient2_SR_1_IM00005.tif'
        
        Returns dict with:
            patient_id: e.g. 'patient100'
            series_id: e.g. 'SR_2'
            slice_idx: integer index, e.g. 28
            label: 'covid' or 'normal'
        """
        fname = Path(filepath).name
        
        # Extract patient ID (e.g. patient100)
        p_match = re.search(r'(patient\d+)', fname, re.IGNORECASE)
        patient_id = p_match.group(1).lower() if p_match else "unknown"

        # Extract series ID (e.g. SR_2, SR_3)
        s_match = re.search(r'(SR_\d+)', fname, re.IGNORECASE)
        series_id = s_match.group(1).upper() if s_match else "SR_DEFAULT"

        # Extract slice index (e.g. IM00028 -> 28)
        i_match = re.search(r'IM(\d+)', fname, re.IGNORECASE)
        slice_idx = int(i_match.group(1)) if i_match else 0

        # Extract label
        label = "covid" if "covid" in fname.lower() else "normal"

        return {
            "filepath": filepath,
            "patient_id": patient_id,
            "series_id": series_id,
            "slice_idx": slice_idx,
            "label": label
        }

    @classmethod
    def generate_triplets(cls, filepaths: List[str]) -> List[Tuple[str, str, str, str]]:
        """
        Groups slices by (patient_id, series_id), sorts by slice_idx,
        and generates (Z-1, Z, Z+1, label) triplets using edge replication.
        
        Returns:
            List of tuples: (path_prev, path_curr, path_next, label)
        """
        # 1. Group slices by scan sequence: (patient_id, series_id)
        scan_groups: Dict[Tuple[str, str], List[Dict]] = {}
        for fp in filepaths:
            info = cls.parse_filename(fp)
            key = (info["patient_id"], info["series_id"])
            if key not in scan_groups:
                scan_groups[key] = []
            scan_groups[key].append(info)

        triplets = []

        # 2. Iterate each isolated scan group
        for (patient_id, series_id), slices in scan_groups.items():
            # Sort strictly by slice index
            sorted_slices = sorted(slices, key=lambda s: s["slice_idx"])
            n = len(sorted_slices)
            
            if n == 0:
                continue
            elif n == 1:
                # Single slice scan: replicate for all 3 channels
                curr = sorted_slices[0]["filepath"]
                triplets.append((curr, curr, curr, sorted_slices[0]["label"]))
                continue

            for i in range(n):
                # Clamp boundary handling:
                # If i == 0: prev is slice 0 (replicated)
                # If i == n-1: next is slice n-1 (replicated)
                prev_idx = max(0, i - 1)
                curr_idx = i
                next_idx = min(n - 1, i + 1)

                p_prev = sorted_slices[prev_idx]["filepath"]
                p_curr = sorted_slices[curr_idx]["filepath"]
                p_next = sorted_slices[next_idx]["filepath"]
                lbl = sorted_slices[curr_idx]["label"]

                triplets.append((p_prev, p_curr, p_next, lbl))

        return triplets

