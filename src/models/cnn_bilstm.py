"""
PulmoVision — Stage 2 Model Architectures
Modular sequence architectures for CT patient classification:
- Experiment A: EfficientNet-B0 features + Global Mean Pooling + MLP
- Experiment B: EfficientNet-B0 features + BiLSTM + Global Mean Pooling + MLP
- Experiment C: EfficientNet-B0 features + BiLSTM + Additive Attention + MLP
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PulmoGlobalMeanPooling(nn.Module):
    """
    Experiment A: Atemporal baseline.
    Computes masked global mean pooling over slice embeddings, followed by an MLP head.
    
    Args:
        input_dim: Embedding dimension from backbone (default: 1280).
        hidden_dim: MLP projection dimension (default: 64).
        num_classes: Output categories (default: 2).
        dropout: Classifier dropout rate (default: 0.3).
    """
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor [B, T_max, 1280]
            mask: Optional boolean/float mask [B, T_max] (1.0 = valid, 0.0 = padded)
            lengths: Optional lengths tensor [B]
        Returns:
            logits: [B, num_classes]
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) # [B, T_max, 1]
            sum_x = torch.sum(x * mask_expanded, dim=1) # [B, 1280]
            count = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0) # [B, 1]
            context = sum_x / count # [B, 1280]
        else:
            context = torch.mean(x, dim=1)
            
        return self.classifier(context)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class PulmoBiLSTMMeanPooling(nn.Module):
    """
    Experiment B: Sequence recurrence baseline.
    Encodes slice embeddings via BiLSTM, applies masked global mean pooling over
    hidden states, followed by an MLP head.
    
    Args:
        input_dim: Embedding dimension (default: 1280).
        hidden_dim: BiLSTM hidden size per direction (default: 128 -> 2H=256).
        mlp_dim: MLP projection dimension (default: 64).
        num_classes: Output categories (default: 2).
        dropout: Classifier dropout rate (default: 0.3).
    """
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 128,
        mlp_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor [B, T_max, 1280]
            mask: [B, T_max] (1.0 = valid, 0.0 = padded)
            lengths: Lengths tensor [B]
        Returns:
            logits: [B, num_classes]
        """
        if lengths is not None:
            # Use packed sequences to bypass recurrent steps on padded tokens
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            h_seq, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1)) # [B, T_max, 2H]
        else:
            h_seq, _ = self.lstm(x)
            
        if mask is not None:
            # Mask out any padded positions
            mask_exp = mask.unsqueeze(-1) # [B, T_max, 1]
            sum_h = torch.sum(h_seq * mask_exp, dim=1) # [B, 2H]
            count = torch.clamp(torch.sum(mask_exp, dim=1), min=1.0)
            context = sum_h / count # [B, 2H]
        else:
            context = torch.mean(h_seq, dim=1)
            
        return self.classifier(context)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class PulmoBiLSTMAttention(nn.Module):
    """
    Experiment C: Proposed Champion Architecture.
    Encodes slice embeddings via BiLSTM, aggregates via masked Additive Self-Attention
    Pooling (Bahdanau-style), and classifies via an MLP head.
    
    Args:
        input_dim: Embedding dimension (default: 1280).
        hidden_dim: BiLSTM hidden size per direction (default: 128 -> 2H=256).
        attn_dim: Internal attention projection dimension (default: 64).
        mlp_dim: MLP projection dimension (default: 64).
        num_classes: Output categories (default: 2).
        dropout: Classifier dropout rate (default: 0.3).
    """
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 128,
        attn_dim: int = 64,
        mlp_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Additive attention components: a(h) = v_a^T tanh(W_a h + b_a)
        self.w_a = nn.Linear(hidden_dim * 2, attn_dim, bias=True)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Sequence tensor [B, T_max, 1280]
            mask: [B, T_max] (1.0 = valid, 0.0 = padded)
            lengths: Lengths tensor [B]
            return_attention: If True, returns (logits, attention_weights)
        Returns:
            logits: [B, num_classes]
            attention_weights: [B, T_max] or None
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            h_seq, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1)) # [B, T_max, 2H]
        else:
            h_seq, _ = self.lstm(x)

        # 1. Attention Energy Scores: u_t = tanh(W_a h_t)
        u_t = torch.tanh(self.w_a(h_seq)) # [B, T_max, attn_dim]
        scores = self.v_a(u_t).squeeze(-1) # [B, T_max]

        # 2. Masking: set padded tokens to -1e9 so softmax produces 0 weight
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Normalized attention weights: alpha_t
        alpha = torch.softmax(scores, dim=1) # [B, T_max]

        # 4. Context Vector: c = sum_t (alpha_t * h_t)
        alpha_exp = alpha.unsqueeze(-1) # [B, T_max, 1]
        context = torch.sum(alpha_exp * h_seq, dim=1) # [B, 2H]

        logits = self.classifier(context)

        if return_attention:
            return logits, alpha
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def get_stage2_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function for Stage 2 models.
    Valid names: 'effnet_b0_mean_pooling', 'effnet_b0_bilstm_mean', 'effnet_b0_bilstm_attention'
    """
    registry = {
        "effnet_b0_mean_pooling": PulmoGlobalMeanPooling,
        "effnet_b0_bilstm_mean": PulmoBiLSTMMeanPooling,
        "effnet_b0_bilstm_attention": PulmoBiLSTMAttention,
    }
    if model_name not in registry:
        raise ValueError(f"Unknown Stage 2 model: '{model_name}'. Available: {list(registry.keys())}")
    return registry[model_name](**kwargs)
