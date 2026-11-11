"""
PulmoVision — Regression Tests for Equal Patient-Level Optimization
Verifies:
1. Patient with 1 series vs patient with 3 series.
2. Equal patient-level loss contribution (each patient contributes exactly 1/B).
3. Invariance to series ordering within multi-series patients.
4. Correct gradient propagation through patient-balanced loss.
5. Strict absence of cross-patient aggregation leakage.
6. Multi-series patients in single batch and across batches.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sequence_dataset import (
    PatientSequenceDataset,
    collate_patient_balanced_batch,
    CachedEmbeddingDataset,
    collate_sequence_embeddings,
)
from src.evaluation.nested_cv import (
    compute_patient_balanced_loss,
    train_epoch,
    evaluate_patient_predictions,
)
from src.models.cnn_bilstm import PulmoGlobalMeanPooling


class TestEqualPatientOptimization:

    @pytest.fixture
    def synthetic_patient_cache(self):
        """
        Creates a synthetic cache structured identically to dev_patient_features_effnet_b0.pt:
        - patient_single: 1 series (SR_1, 5 slices)
        - patient_multi: 3 series (SR_1, SR_2, SR_3, 4 slices each)
        - patient_pair: 2 series (SR_1, SR_2, 3 slices each)
        """
        torch.manual_seed(42)
        patients_dict = {
            "p_single": {
                "patient_id": "p_single",
                "label": 1,
                "label_str": "covid",
                "series_ids": ["SR_1"],
                "series": {
                    "SR_1": {
                        "patient_id": "p_single",
                        "series_id": "SR_1",
                        "label": 1,
                        "num_slices": 5,
                        "slice_indices": [1, 2, 3, 4, 5],
                        "embeddings": torch.randn(5, 1280),
                    }
                },
            },
            "p_multi": {
                "patient_id": "p_multi",
                "label": 0,
                "label_str": "normal",
                "series_ids": ["SR_1", "SR_2", "SR_3"],
                "series": {
                    "SR_1": {
                        "patient_id": "p_multi",
                        "series_id": "SR_1",
                        "label": 0,
                        "num_slices": 4,
                        "slice_indices": [1, 2, 3, 4],
                        "embeddings": torch.randn(4, 1280),
                    },
                    "SR_2": {
                        "patient_id": "p_multi",
                        "series_id": "SR_2",
                        "label": 0,
                        "num_slices": 4,
                        "slice_indices": [1, 2, 3, 4],
                        "embeddings": torch.randn(4, 1280),
                    },
                    "SR_3": {
                        "patient_id": "p_multi",
                        "series_id": "SR_3",
                        "label": 0,
                        "num_slices": 4,
                        "slice_indices": [1, 2, 3, 4],
                        "embeddings": torch.randn(4, 1280),
                    },
                },
            },
            "p_pair": {
                "patient_id": "p_pair",
                "label": 1,
                "label_str": "covid",
                "series_ids": ["SR_1", "SR_2"],
                "series": {
                    "SR_1": {
                        "patient_id": "p_pair",
                        "series_id": "SR_1",
                        "label": 1,
                        "num_slices": 3,
                        "slice_indices": [1, 2, 3],
                        "embeddings": torch.randn(3, 1280),
                    },
                    "SR_2": {
                        "patient_id": "p_pair",
                        "series_id": "SR_2",
                        "label": 1,
                        "num_slices": 3,
                        "slice_indices": [1, 2, 3],
                        "embeddings": torch.randn(3, 1280),
                    },
                },
            },
        }
        return {"patients": patients_dict}

    def test_patient_sequence_dataset_structure(self, synthetic_patient_cache):
        """Verifies PatientSequenceDataset indexes strictly by patient."""
        ds = PatientSequenceDataset(synthetic_patient_cache)
        assert len(ds) == 3
        p_single = [ds[i] for i in range(len(ds)) if ds[i]["patient_id"] == "p_single"][0]
        assert p_single["num_series"] == 1

        p_multi = [ds[i] for i in range(len(ds)) if ds[i]["patient_id"] == "p_multi"][0]
        assert p_multi["num_series"] == 3

    def test_collate_patient_balanced_batch_shapes(self, synthetic_patient_cache):
        """Verifies collate_patient_balanced_batch produces correct mappings."""
        ds = PatientSequenceDataset(synthetic_patient_cache)
        loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate_patient_balanced_batch)
        batch = next(iter(loader))

        assert batch["embeddings"].shape == (6, 5, 1280) # 3 + 2 + 1 = 6 series, max T = 5
        assert batch["mask"].shape == (6, 5)
        # Patient list is sorted alphabetically: p_multi (3 series, len 4), p_pair (2 series, len 3), p_single (1 series, len 5)
        assert batch["lengths"].tolist() == [4, 4, 4, 3, 3, 5]
        assert batch["series_patient_indices"].tolist() == [0, 0, 0, 1, 1, 2] # 0: multi, 1: pair, 2: single
        assert batch["patient_labels"].shape == (3,)
        assert batch["num_series_per_patient"].tolist() == [3, 2, 1]
        assert len(batch["patient_ids"]) == 3

    def test_equal_patient_level_loss_contribution(self):
        """
        Mathematical proof test:
        Verifies that a patient with 1 series and a patient with 3 series
        have strictly equal influence on the total batch loss.
        """
        # Batch: B = 2 patients (Patient 0: 1 series, Patient 1: 3 series) -> M = 4 series total
        series_patient_indices = torch.tensor([0, 1, 1, 1], dtype=torch.long)
        patient_labels = torch.tensor([1, 0], dtype=torch.long)
        criterion = nn.CrossEntropyLoss(reduction="none")

        logits = torch.tensor([
            [0.0, 2.0], # Patient 0, series 1 (true: 1)
            [2.0, 0.0], # Patient 1, series 1 (true: 0)
            [2.0, 0.0], # Patient 1, series 2 (true: 0)
            [2.0, 0.0], # Patient 1, series 3 (true: 0)
        ], requires_grad=True)

        batch_loss, p_losses = compute_patient_balanced_loss(
            logits, series_patient_indices, patient_labels, criterion
        )

        # Both patients must have exactly equal patient loss
        assert p_losses[0].item() == pytest.approx(p_losses[1].item(), abs=1e-5)
        # Batch loss is the unweighted mean (0.5 * p_loss[0] + 0.5 * p_loss[1])
        assert batch_loss.item() == pytest.approx(p_losses[0].item(), abs=1e-5)

        # Gradient verification:
        batch_loss.backward()

        # The gradient of batch_loss with respect to Patient 0's series logit is 0.5 * d(L)/d(logits)
        # The sum of gradients across Patient 1's 3 series logits must EXACTLY equal Patient 0's gradient norm
        grad_p0 = logits.grad[0].norm().item()
        grad_p1_total = (logits.grad[1] + logits.grad[2] + logits.grad[3]).norm().item()
        assert grad_p0 == pytest.approx(grad_p1_total, abs=1e-5), "Total gradient influence must be equal"

        # Each individual series of Patient 1 receives exactly 1/3 the gradient of Patient 0's series
        for s_idx in [1, 2, 3]:
            assert logits.grad[s_idx].norm().item() == pytest.approx(grad_p0 / 3.0, abs=1e-5)

    def test_invariance_to_series_ordering(self):
        """Verifies that reordering the series of a patient yields identical patient loss and gradients."""
        series_patient_indices = torch.tensor([0, 1, 1, 1], dtype=torch.long)
        patient_labels = torch.tensor([1, 0], dtype=torch.long)
        criterion = nn.CrossEntropyLoss(reduction="none")

        logits_orig = torch.tensor([
            [0.5, 1.5],
            [1.0, 0.2],
            [2.5, 0.1],
            [0.8, 0.4],
        ], requires_grad=True)

        loss_orig, _ = compute_patient_balanced_loss(logits_orig, series_patient_indices, patient_labels, criterion)
        loss_orig.backward()

        # Permute series 1, 2, 3 of patient 1 (rows 1, 2, 3 -> rows 3, 1, 2)
        logits_perm = torch.tensor([
            [0.5, 1.5],
            [0.8, 0.4],
            [1.0, 0.2],
            [2.5, 0.1],
        ], requires_grad=True)

        loss_perm, _ = compute_patient_balanced_loss(logits_perm, series_patient_indices, patient_labels, criterion)
        loss_perm.backward()

        assert loss_orig.item() == pytest.approx(loss_perm.item(), abs=1e-6)
        assert logits_orig.grad[0].tolist() == pytest.approx(logits_perm.grad[0].tolist(), abs=1e-6)

    def test_no_cross_patient_leakage(self):
        """Verifies modifying one patient's features or loss has zero effect on another patient's gradients."""
        series_patient_indices = torch.tensor([0, 1, 1], dtype=torch.long)
        patient_labels = torch.tensor([1, 0], dtype=torch.long)
        criterion = nn.CrossEntropyLoss(reduction="none")

        logits_a = torch.tensor([[1.0, 0.5], [2.0, 0.1], [1.5, 0.3]], requires_grad=True)
        loss_a, _ = compute_patient_balanced_loss(logits_a, series_patient_indices, patient_labels, criterion)
        loss_a.backward()
        grad_p1_before = logits_a.grad[1:].clone()

        # Change Patient 0 logits completely
        logits_b = torch.tensor([[99.0, -99.0], [2.0, 0.1], [1.5, 0.3]], requires_grad=True)
        loss_b, _ = compute_patient_balanced_loss(logits_b, series_patient_indices, patient_labels, criterion)
        loss_b.backward()
        grad_p1_after = logits_b.grad[1:].clone()

        # Patient 1 gradients must be bit-for-bit identical despite Patient 0's massive logit shift
        assert torch.allclose(grad_p1_before, grad_p1_after, atol=1e-6)

    def test_patient_balanced_train_epoch_and_eval(self, synthetic_patient_cache):
        """Tests end-to-end patient-balanced train_epoch and evaluate_patient_predictions."""
        ds = PatientSequenceDataset(synthetic_patient_cache)
        loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_patient_balanced_batch)

        model = PulmoGlobalMeanPooling()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cpu")

        # Train one epoch
        loss_epoch = train_epoch(model, loader, optimizer, criterion, device)
        assert loss_epoch > 0.0
        assert not torch.isnan(torch.tensor(loss_epoch))

        # Evaluate predictions
        eval_loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_patient_balanced_batch)
        preds = evaluate_patient_predictions(model, eval_loader, device)

        assert len(preds) == 3
        for pid in ["p_single", "p_multi", "p_pair"]:
            assert pid in preds
            assert 0.0 <= preds[pid]["pred_prob"] <= 1.0
            assert preds[pid]["num_series"] == (1 if pid == "p_single" else (3 if pid == "p_multi" else 2))
