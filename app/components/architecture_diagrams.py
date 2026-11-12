"""
PulmoVision — Model Architecture Diagrams & Inspector
Renders structural diagrams for Stage 1 CNNs and Stage 2 Sequence Architectures (A, B, C).
Clearly displays dimensions, parameter counts, and layer descriptions.
"""

from typing import Dict, Any
import streamlit as st


def render_stage2_architecture_diagram(model_key: str = "B") -> None:
    """
    Renders visual structural flowchart for Stage 2 architectures.
    """
    if model_key == "A":
        st.markdown(
            """
            <div class="pv-card" style="border-left: 4px solid #64748b;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: var(--pv-text-primary);">Experiment A: Atemporal Global Mean Pooling</h4>
                    <span style="font-family: var(--pv-font-mono); font-size: 0.8rem; background: var(--pv-bg-elevated); padding: 3px 8px; border-radius: 4px;">
                        82,114 trainable params
                    </span>
                </div>
                <p style="font-size: 0.88rem; color: var(--pv-text-secondary); margin-bottom: 14px;">
                    Atemporal baseline computing uniform arithmetic averaging across slice embeddings without temporal sequence recurrence.
                </p>
                <div style="background-color: var(--pv-bg-app); border: 1px solid var(--pv-border); border-radius: 6px; padding: 16px; font-family: var(--pv-font-mono); font-size: 0.82rem; line-height: 1.6;">
                    <div style="color: var(--pv-text-muted);">INPUT: Ordered CT Slices [B, T_max, 512, 512]</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Preprocessing &bull; Clip [200, 1800] &bull; Scale [0, 1] &bull; Resize 224&times;224</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; Frozen ImageNet EfficientNet-B0 Backbone (zero in-domain exposure)</div>
                    <div style="color: var(--pv-accent-primary);">EMBEDDINGS: Tensor e_{p,t} &in; R^{B &times; T_max &times; 1280} + Binary Padding Mask</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; <strong>Masked Global Mean Pooling over T:</strong> c_p = (1/T_p) &Sigma; e_{p,t} &in; R^{B &times; 1280}</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Projection: Linear(1280 &rarr; 64) &bull; SiLU &bull; Dropout(0.3)</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Classification: Linear(64 &rarr; 2)</div>
                    <div style="color: var(--pv-success);">OUTPUT: Patient-Level Logits [B, 2] &rarr; Softmax Score &rarr; Decision at &tau; = 0.50</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif model_key == "B":
        st.markdown(
            """
            <div class="pv-card" style="border-left: 4px solid var(--pv-accent-primary);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: var(--pv-text-primary);">Experiment B: EfficientNet-B0 + BiLSTM + Mean Pooling 🏆</h4>
                    <span style="font-family: var(--pv-font-mono); font-size: 0.8rem; background: var(--pv-bg-elevated); padding: 3px 8px; border-radius: 4px; color: var(--pv-accent-primary);">
                        1,460,418 trainable params
                    </span>
                </div>
                <p style="font-size: 0.88rem; color: var(--pv-text-secondary); margin-bottom: 14px;">
                    Leading sequence architecture capturing bidirectional contextual memory along the craniocaudal axial axis.
                </p>
                <div style="background-color: var(--pv-bg-app); border: 1px solid var(--pv-border); border-radius: 6px; padding: 16px; font-family: var(--pv-font-mono); font-size: 0.82rem; line-height: 1.6;">
                    <div style="color: var(--pv-text-muted);">INPUT: Ordered CT Slices [B, T_max, 512, 512]</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Preprocessing &bull; Clip [200, 1800] &bull; Scale [0, 1] &bull; Resize 224&times;224</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; Frozen ImageNet EfficientNet-B0 Backbone (1280-d embeddings)</div>
                    <div style="color: var(--pv-accent-primary);">EMBEDDINGS: Tensor e_{p,t} &in; R^{B &times; T_max &times; 1280} + Binary Padding Mask</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; <strong>1-Layer Bidirectional LSTM:</strong> hidden_dim=128 per direction &rarr; States h_t &in; R^{B &times; T_max &times; 256}</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; <strong>Masked Global Mean Pooling over Recurrent States:</strong> c_p = (1/T_p) &Sigma; h_{p,t} &in; R^{B &times; 256}</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Projection: Linear(256 &rarr; 64) &bull; SiLU &bull; Dropout(0.3)</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Classification: Linear(64 &rarr; 2)</div>
                    <div style="color: var(--pv-success);">OUTPUT: Patient-Level Logits [B, 2] &rarr; Softmax Score &rarr; Decision at &tau; = 0.50</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif model_key == "C":
        st.markdown(
            """
            <div class="pv-card" style="border-left: 4px solid var(--pv-warning);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: var(--pv-text-primary);">Experiment C: EfficientNet-B0 + BiLSTM + Additive Attention</h4>
                    <span style="font-family: var(--pv-font-mono); font-size: 0.8rem; background: var(--pv-bg-elevated); padding: 3px 8px; border-radius: 4px; color: var(--pv-warning);">
                        1,476,930 trainable params
                    </span>
                </div>
                <p style="font-size: 0.88rem; color: var(--pv-text-secondary); margin-bottom: 14px;">
                    Recurrent architecture with single-head Bahdanau additive attention learning dynamic slice saliency weights.
                </p>
                <div style="background-color: var(--pv-bg-app); border: 1px solid var(--pv-border); border-radius: 6px; padding: 16px; font-family: var(--pv-font-mono); font-size: 0.82rem; line-height: 1.6;">
                    <div style="color: var(--pv-text-muted);">INPUT: Ordered CT Slices [B, T_max, 512, 512]</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Preprocessing &bull; Clip [200, 1800] &bull; Scale [0, 1] &bull; Resize 224&times;224</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; Frozen ImageNet EfficientNet-B0 Backbone (1280-d embeddings)</div>
                    <div style="color: var(--pv-accent-primary);">EMBEDDINGS: Tensor e_{p,t} &in; R^{B &times; T_max &times; 1280} + Binary Padding Mask</div>
                    <div style="color: var(--pv-text-primary); margin-left: 20px;">&darr; <strong>1-Layer Bidirectional LSTM:</strong> Recurrent states h_t &in; R^{B &times; T_max &times; 256}</div>
                    <div style="color: var(--pv-warning); margin-left: 20px;">&darr; <strong>Additive Attention Head:</strong> u_t = tanh(W_a h_t + b_a) &in; R^64 &bull; e_t = v_a^T u_t &bull; &alpha;_t = softmax(e_t)</div>
                    <div style="color: var(--pv-warning); margin-left: 20px;">&darr; <strong>Context Vector Aggregation:</strong> c_p = &Sigma; &alpha;_t h_{p,t} &in; R^{B &times; 256}</div>
                    <div style="color: var(--pv-accent-secondary); margin-left: 20px;">&darr; Projection: Linear(256 &rarr; 64) &bull; SiLU &bull; Dropout(0.3) &bull; Linear(64 &rarr; 2)</div>
                    <div style="color: var(--pv-success);">OUTPUT: Patient-Level Logits [B, 2] &rarr; Softmax Score &rarr; Decision at &tau; = 0.50 (0.30 in Fold 5)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_component_inspector(component_name: str) -> None:
    """Displays detailed technical documentation for selected architectural component."""
    components = {
        "BiLSTM Encoder": {
            "what": "1-layer Bidirectional Long Short-Term Memory network.",
            "why": "Processes ordered axial slices forward (apex to base) and backward (base to apex) to retain context across contiguous anatomical cross-sections.",
            "in_dim": "[B, T, 1280] (Slice embeddings from frozen EfficientNet-B0)",
            "out_dim": "[B, T, 256] (Concatenated forward and backward hidden states)",
            "params": "1,443,840 parameters (4 * 128 * (1280 + 128 + 2) * 2 directions)"
        },
        "Masked Global Mean Pooling": {
            "what": "Sequence length-aware arithmetic mean pooling across time dimension T.",
            "why": "Aggregates variable-length sequences into a fixed-size patient representation while ignoring zero-padded tokens.",
            "in_dim": "[B, T, 256] + mask [B, T]",
            "out_dim": "[B, 256] (Patient-level sequence representation)",
            "params": "0 parameters (non-parametric pooling operation)"
        },
        "Bahdanau Additive Attention": {
            "what": "Single-head learnable attention mechanism (energy projection + context sum).",
            "why": "Learns slice-specific importance weights alpha_t, hypothesizing that focal lesion slices contribute more than normal parenchymal slices.",
            "in_dim": "[B, T, 256] + mask [B, T]",
            "out_dim": "[B, 256] (Weighted context vector c_p)",
            "params": "16,512 parameters (Linear(256 -> 64) with bias + Linear(64 -> 1, no bias))"
        },
        "MLP Classifier": {
            "what": "2-layer classification head with SiLU activation and dropout regularization.",
            "why": "Maps pooled patient representation to class logits (Normal vs COVID-19).",
            "in_dim": "[B, 256] (or [B, 1280] in Exp A)",
            "out_dim": "[B, 2] (Raw patient-level diagnostic logits)",
            "params": "16,578 parameters (Linear(256 -> 64) + bias: 16,448; Linear(64 -> 2) + bias: 130)"
        }
    }

    info = components.get(component_name, components["BiLSTM Encoder"])
    st.markdown(
        f"""
        <div class="pv-card" style="background-color: var(--pv-bg-elevated); margin-top: 14px;">
            <div style="font-weight: 600; font-size: 0.95rem; color: var(--pv-accent-primary); margin-bottom: 6px;">
                Component Deep-Dive: {component_name}
            </div>
            <div style="font-size: 0.86rem; color: var(--pv-text-primary); margin-bottom: 6px;">
                <strong>Function:</strong> {info['what']}
            </div>
            <div style="font-size: 0.86rem; color: var(--pv-text-secondary); margin-bottom: 8px;">
                <strong>Scientific Rationale:</strong> {info['why']}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: var(--pv-font-mono); font-size: 0.78rem; border-top: 1px solid var(--pv-border); padding-top: 8px;">
                <div><span style="color: var(--pv-text-muted);">INPUT:</span> {info['in_dim']}</div>
                <div><span style="color: var(--pv-text-muted);">OUTPUT:</span> {info['out_dim']}</div>
                <div style="grid-column: span 2;"><span style="color: var(--pv-text-muted);">PARAMETERS:</span> {info['params']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

