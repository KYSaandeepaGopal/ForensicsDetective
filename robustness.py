import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
AUG_DIR     = BASE_DIR / "augmented_images"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LABELS = {"word": 0, "google": 1, "python": 2}

CONDITIONS = [
    "original",
    "aug1_noise",
    "aug2_jpeg",
    "aug3_dpi",
    "aug4_crop",
    "aug5_bitdepth"
]

IMG_SIZE = (64, 64)

# ── Load images for one condition ──────────────────────────────────────────
def load_condition(condition):
    X, y = [], []
    for label_name, label_id in LABELS.items():
        folder  = AUG_DIR / label_name
        pattern = f"*_{condition}.png"
        files   = sorted(folder.glob(pattern))
        print(f"  [{label_name}] {condition}: {len(files)} files found")
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten())
            y.append(label_id)
    return np.array(X), np.array(y)

# ── Main ───────────────────────────────────────────────────────────────────
def run_robustness():

    # Step 1 — Load original images and train
    print("\n── Step 1: Loading original images ──────────────")
    X_orig, y_orig = load_condition("original")
    print(f"  Total: {len(X_orig)} images loaded")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_orig,
        test_size=0.2,
        random_state=42,
        stratify=y_orig
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 2 — Train SVM and SGD on original data only
    print("\n── Step 2: Training classifiers on original data ─")
    classifiers = {
        "SVM" : SVC(kernel="rbf", C=1.0, random_state=42),
        "SGD" : SGDClassifier(max_iter=1000, random_state=42),
    }
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(X_train, y_train)
        print(f"  {name} trained.")

    # Step 3 — Test on original + each augmentation
    print("\n── Step 3: Evaluating on all conditions ──────────")
    results = {name: {} for name in classifiers}

    for condition in CONDITIONS:
        print(f"\n  Condition: {condition}")
        if condition == "original":
            X_eval = X_test
            y_eval = y_test
        else:
            X_cond, y_cond = load_condition(condition)
            X_eval = scaler.transform(X_cond)
            y_eval = y_cond

        for name, clf in classifiers.items():
            acc = accuracy_score(y_eval, clf.predict(X_eval))
            results[name][condition] = round(acc * 100, 2)
            print(f"    {name}: {results[name][condition]}%")

    # Step 4 — Save CSV
    df = pd.DataFrame(results).T
    df.columns = ["Original", "Noise", "JPEG", "DPI", "Crop", "Bitdepth"]
    df.index.name = "Classifier"
    csv_path = RESULTS_DIR / "robustness_results.csv"
    df.to_csv(csv_path)
    print(f"\n✓ Results saved → {csv_path}")
    print(df.to_string())

    # Step 5 — Plot robustness curves
    plot_curves(results)
    plot_bar(results)

# ── Plots ──────────────────────────────────────────────────────────────────
def plot_curves(results):
    condition_labels = ["Original", "Noise", "JPEG", "DPI", "Crop", "Bitdepth"]
    plt.figure(figsize=(10, 6))
    for clf_name, cond_results in results.items():
        y_vals = [cond_results.get(c, 0) for c in CONDITIONS]
        plt.plot(condition_labels, y_vals, marker='o', linewidth=2, label=clf_name)
    plt.title("Robustness Curves — Accuracy Drop per Augmentation (SVM vs SGD)")
    plt.xlabel("Test Condition")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "robustness_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Robustness curve saved → {path}")

def plot_bar(results):
    condition_labels = ["Original", "Noise", "JPEG", "DPI", "Crop", "Bitdepth"]
    df = pd.DataFrame(results).T
    df.columns = condition_labels
    df.plot(kind="bar", figsize=(10, 6), width=0.6)
    plt.title("Accuracy per Condition — SVM vs SGD")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=0)
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    path = RESULTS_DIR / "accuracy_bar_chart.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Bar chart saved → {path}")

if __name__ == "__main__":
    run_robustness()