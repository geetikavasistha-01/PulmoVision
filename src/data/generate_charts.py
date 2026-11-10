"""
Generate visual charts and flow summary for PulmoVision data & pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def generate_visuals():
    out_dir = "docs/data_and_pipeline_overview"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Chart: Patient Class Distribution
    patient_csv = "data/metadata/Annotation/CSV/Patient_details.csv"
    if os.path.exists(patient_csv):
        df = pd.read_csv(patient_csv)
        counts = df['COVID-19 Infection'].value_counts()
        
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(counts.index, counts.values, color=['#3498db', '#e74c3c'], width=0.5, edgecolor='black', linewidth=1.2)
        ax.set_title("COVID-CTset Patient-Level Distribution (N = 377)", fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel("Number of Patients", fontsize=11)
        ax.set_ylim(0, 320)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            percentage = (height / len(df)) * 100
            ax.annotate(f'{height} ({percentage:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        chart_path = os.path.join(out_dir, "patient_distribution.png")
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"Saved: {chart_path}")

    # 2. Chart: 16-Bit CT Sample & Intensity Histogram
    sample_img_path = "data/interim/sample_audit/TrainValidation/137covid_patient100_SR_2_IM00028.tif"
    if os.path.exists(sample_img_path):
        img = Image.open(sample_img_path)
        arr = np.array(img, dtype=np.float32)
        
        # Windowed version for lung comparison: window [200, 1800]
        window_min, window_max = 200, 1800
        windowed = np.clip(arr, window_min, window_max)
        windowed = (windowed - window_min) / (window_max - window_min)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Raw normalized
        norm_raw = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        ax1.imshow(norm_raw, cmap='gray')
        ax1.set_title("Raw 16-Bit Slice\n(Linear min-max normalized)", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Windowed contrast
        ax2.imshow(windowed, cmap='bone')
        ax2.set_title("Windowed CT Slice\n(Lung Parenchyma Window)", fontsize=11, fontweight='bold')
        ax2.axis('off')

        # Histogram
        ax3.hist(arr.ravel(), bins=60, color='#2c3e50', edgecolor='white', alpha=0.85)
        ax3.axvline(window_min, color='#e67e22', linestyle='--', linewidth=1.5, label=f'Window Min ({window_min})')
        ax3.axvline(window_max, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Window Max ({window_max})')
        ax3.set_title("16-Bit Pixel Intensity Distribution", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Voxel Radiodensity Intensity (uint16)", fontsize=10)
        ax3.set_ylabel("Voxel Count", fontsize=10)
        ax3.set_yscale('log')
        ax3.grid(True, linestyle=':', alpha=0.5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        ct_chart_path = os.path.join(out_dir, "ct_intensity_and_slice_analysis.png")
        plt.savefig(ct_chart_path, dpi=200)
        plt.close()
        print(f"Saved: {ct_chart_path}")

    # 3. Chart: 2.5D Architecture Visual Diagram
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')

    boxes = [
        ("Slice Z-1\n[512, 512]", 0.08, 0.70, '#ecf0f1'),
        ("Slice Z\n[512, 512]", 0.08, 0.45, '#bdc3c7'),
        ("Slice Z+1\n[512, 512]", 0.08, 0.20, '#ecf0f1'),
        ("2.5D Channel\nStacking Engine\nTensor: [3, 224, 224]", 0.38, 0.45, '#3498db'),
        ("Pretrained\nBackbone\n(EfficientNet-B0)", 0.65, 0.45, '#2ecc71'),
        ("Prediction\n& Grad-CAM", 0.90, 0.45, '#9b59b6')
    ]

    for label, x, y, col in boxes:
        ax.text(x, y, label, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=col, edgecolor="black", linewidth=1.5),
                fontsize=10, fontweight='bold', color="black" if col != '#3498db' and col != '#9b59b6' else "white")

    # Arrows
    arrow_props = dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, shrink=0.1)
    ax.annotate('', xy=(0.28, 0.50), xytext=(0.17, 0.70), arrowprops=arrow_props)
    ax.annotate('', xy=(0.28, 0.45), xytext=(0.17, 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(0.28, 0.40), xytext=(0.17, 0.20), arrowprops=arrow_props)
    ax.annotate('', xy=(0.54, 0.45), xytext=(0.48, 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(0.81, 0.45), xytext=(0.75, 0.45), arrowprops=arrow_props)

    ax.set_title("PulmoVision 2.5D Input Stacking & Inference Pipeline Flow", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    flow_path = os.path.join(out_dir, "pipeline_architecture_flow.png")
    plt.savefig(flow_path, dpi=200)
    plt.close()
    print(f"Saved: {flow_path}")

if __name__ == "__main__":
    generate_visuals()

