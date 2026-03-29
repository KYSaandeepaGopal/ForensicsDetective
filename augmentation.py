import cv2
import numpy as np
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Paths (all relative to repo root) ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIRS = {
    "word"   : BASE_DIR / "word_pdfs_png",
    "google" : BASE_DIR / "google_docs_pdfs_png",
    "python" : BASE_DIR / "python_pdfs_png",
}

OUT_DIR = BASE_DIR / "augmented_images"

# ── 5 Augmentation Functions ───────────────────────────────────────────────
def gaussian_noise(img):
    sigma = random.uniform(5, 20)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def jpeg_compression(img):
    quality = random.randint(20, 80)
    _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

def dpi_downsample(img):
    h, w = img.shape[:2]
    scale = 0.24  # fixed 72 DPI
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def random_crop(img):
    h, w = img.shape[:2]
    pct = random.uniform(0.01, 0.03)
    t, b = int(h * pct), int(h * (1 - pct))
    l, r = int(w * pct), int(w * (1 - pct))
    return cv2.resize(img[t:b, l:r], (w, h), interpolation=cv2.INTER_LINEAR)

def bit_depth_reduction(img):
    return ((img // 16) * 16).astype(np.uint8)

AUGMENTATIONS = {
    "aug1_noise"    : gaussian_noise,
    "aug2_jpeg"     : jpeg_compression,
    "aug3_dpi"      : dpi_downsample,
    "aug4_crop"     : random_crop,
    "aug5_bitdepth" : bit_depth_reduction,
}

# ── Main ───────────────────────────────────────────────────────────────────
def augment_dataset():
    total = 0
    for label, src_dir in SOURCE_DIRS.items():
        if not src_dir.exists():
            print(f"[WARN] Not found: {src_dir}")
            continue

        out_label_dir = OUT_DIR / label
        out_label_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(src_dir.glob("*.png"))
        print(f"\n[{label}] {len(images)} images → {len(images)*6} expected")

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [SKIP] {img_path.name}")
                continue
            stem = img_path.stem
            cv2.imwrite(str(out_label_dir / f"{stem}_original.png"), img)
            for aug_name, aug_fn in AUGMENTATIONS.items():
                cv2.imwrite(str(out_label_dir / f"{stem}_{aug_name}.png"), aug_fn(img))
            total += 6

    print(f"\n✓ Done — {total} images saved to {OUT_DIR}")

    # Verify counts
    print("\n── Verification ──")
    for label, expected_orig in [("word", 398), ("google", 396), ("python", 100)]:
        actual = len(list((OUT_DIR / label).glob("*.png")))
        expected = expected_orig * 6
        status = "✓" if actual == expected else "✗ MISMATCH"
        print(f"  {status} {label}: {actual} / {expected}")

if __name__ == "__main__":
    augment_dataset()