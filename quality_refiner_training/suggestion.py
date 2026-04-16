import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]

def load_data(high_scores_path, low_scores_path):
    """Load high-score and low-score group data."""
    try:
        df_high = pd.read_csv(high_scores_path)
        df_low = pd.read_csv(low_scores_path)
        return df_high, df_low
    except FileNotFoundError:
        print("[ERROR] Could not find G_high_scores.csv or G_low_scores.csv. Ensure they are located in output/ directory.")
        return None, None

def plot_and_analyze_attributes(df_high, df_low):
    """
    For each attribute, plot the KDE (Kernel Density Estimation) distribution
    of G_High and G_Low groups, and mark the P5 and P95 thresholds.
    """
    attribute_score_cols = [f'{name}_score' for name in ATTRIBUTE_NAMES]

    output_dir = 'output/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")

    print("\n========================================")
    print("[INFO] Step 1: Generating Distribution Plots (KDE)")
    print("========================================")

    threshold_data = {}

    for i, attr_col in enumerate(tqdm(attribute_score_cols, desc="Processing Attributes")):
        data_high = df_high[attr_col].dropna()
        data_low = df_low[attr_col].dropna()

        # Threshold calculation
        t_min = data_high.quantile(0.05)
        t_max = data_high.quantile(0.95)
        mu_high = data_high.mean()
        mu_low = data_low.mean()

        threshold_data[attr_col] = {
            'T_min (P5)': f"{t_min:.4f}",
            'T_max (P95)': f"{t_max:.4f}",
            'Ideal_Range_90%': f"[{t_min:.4f}, {t_max:.4f}]",
            'μ_High': f"{mu_high:.4f}",
            'μ_Low': f"{mu_low:.4f}",
        }

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data_high, label=f'G_High (N={len(data_high)})', fill=True, alpha=0.5, color='green')
        sns.kdeplot(data_low, label=f'G_Low (N={len(data_low)})', fill=True, alpha=0.5, color='red')

        plt.axvline(t_min, color='blue', linestyle='--', linewidth=1.5, label=f'T_min (P5)={t_min:.3f}')
        plt.axvline(t_max, color='orange', linestyle='--', linewidth=1.5, label=f'T_max (P95)={t_max:.3f}')
        plt.axvline(mu_high, color='green', linestyle=':', linewidth=1.5, label=f'μ_High={mu_high:.3f}')
        plt.axvline(mu_low, color='red', linestyle=':', linewidth=1.5, label=f'μ_Low={mu_low:.3f}')

        plt.title(f'Distribution Comparison for {attr_col}')
        plt.xlabel('Attribute Score (0 to 1)')
        plt.ylabel('Density')
        plt.xlim(0, 1)
        plt.legend()

        plot_filename = os.path.join(output_dir, f'{attr_col}_kde_comparison.png')
        plt.savefig(plot_filename)
        plt.close()

    print(f"[SUCCESS] All attribute distribution plots saved to {output_dir}/.")

    print("\n========================================")
    print("[INFO] Step 2: Attribute Threshold Analysis Table")
    print("========================================")

    df_thresholds = pd.DataFrame(threshold_data).T
    df_thresholds.index.name = "Attribute"

    print("\n[REPORT] Key Diagnostic Thresholds (based on G_High P5/P95)")
    print("----------------------------------------------------------------------------------------------------------------------")
    print(df_thresholds)
    print("----------------------------------------------------------------------------------------------------------------------")

    print("\n[GUIDE] Interpretation:")
    print("1. T_min (P5): Critical point diagnosing attribute deficiency. Scores below this value significantly reduce aesthetic quality.")
    print("2. T_max (P95): Critical point diagnosing attribute excess. Scores above this value may cause distortion or unnatural appearance.")
    print("3. μ_High/μ_Low: Mean values of high-score and low-score groups. If μ_Low is far from the Ideal Range, the attribute is a key defect in low-score images.")


if __name__ == "__main__":

    # 假设 CSV 文件默认位于 output/ 目录
    HIGH_PATH = os.path.join('output', 'G_high_scores.csv')
    LOW_PATH = os.path.join('output', 'G_low_scores.csv')

    df_high, df_low = load_data(HIGH_PATH, LOW_PATH)

    if df_high is not None and df_low is not None:
        plot_and_analyze_attributes(df_high, df_low)