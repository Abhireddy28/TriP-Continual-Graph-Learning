import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

def compute_metrics(accuracy_matrix):
    """
    Computes Average Accuracy (AA) and Average Forgetting (AF) as defined in Eq. (19):
      AA = 1/T sum_{j=1}^T M_{T, j}
      AF = 1/(T-1) sum_{j=1}^{T-1} (M_{T, j} - M_{j, j})
    """
    T = accuracy_matrix.shape[0]
    aa = np.mean([accuracy_matrix[T - 1, j] for j in range(T)])
    
    if T > 1:
        forgetting = [accuracy_matrix[T - 1, j] - accuracy_matrix[j, j] for j in range(T - 1)]
        af = np.mean(forgetting)
    else:
        af = 0.0

    return aa, af

def plot_performance_matrix(accuracy_matrix, dataset_name="Dataset", save_path="results/matrix.png"):
    """
    Plots the continual learning performance matrix heatmap matching Figure 3 of the TriP paper.
    Each cell (i, j) indicates accuracy on task j after training on task i.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = accuracy_matrix.shape[0]

    mask = np.zeros_like(accuracy_matrix, dtype=bool)
    # Mask out upper triangle since future tasks haven't been trained yet
    for i in range(T):
        for j in range(i + 1, T):
            mask[i, j] = True

    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(
            accuracy_matrix,
            mask=mask,
            annot=(T <= 10),
            fmt=".1f",
            cmap="viridis",
            cbar_kws={'label': 'Accuracy (%)'},
            xticklabels=range(1, T + 1),
            yticklabels=range(1, T + 1)
        )
    else:
        plt.imshow(accuracy_matrix, cmap="viridis")
        plt.colorbar(label='Accuracy (%)')
    plt.title(f"TriP Performance Matrix - {dataset_name}\n(Row: Trained Task, Col: Evaluated Task)")
    plt.xlabel("Evaluated Task $j$")
    plt.ylabel("Trained Task $i$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Performance matrix saved to: {save_path}")
