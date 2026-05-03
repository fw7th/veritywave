import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

# ── supported image extensions ──────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── individual feature extractors ───────────────────────────────────────────


def extract_fft_features(gray_img):
    """
    Analyzes frequency distribution of the image.
    AI images often show unnatural high-frequency energy patterns
    due to upsampling layers in generator architectures.
    """
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))

    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    low_freq_mask = dist_from_center <= radius
    high_freq_mask = ~low_freq_mask

    low_energy = magnitude_spectrum[low_freq_mask].mean()
    high_energy = magnitude_spectrum[high_freq_mask].mean()

    return [
        low_energy,
        high_energy,
        high_energy / (low_energy + 1e-8),
        magnitude_spectrum.std(),
    ]


def extract_noise_features(gray_img):
    """
    Isolates and analyzes the noise residual of the image.
    Real camera noise is random and Gaussian-like.
    AI images have structured or suspiciously clean noise patterns.
    """
    img_float = gray_img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (5, 5), 0)
    noise_residual = img_float - blurred

    flat = noise_residual.flatten()

    variance = np.var(flat)
    skewness = stats.skew(flat)
    kurtosis = stats.kurtosis(flat)

    local_var_map = []
    step = max(gray_img.shape[0] // 4, 1)
    for i in range(0, gray_img.shape[0] - step, step):
        for j in range(0, gray_img.shape[1] - step, step):
            patch = noise_residual[i : i + step, j : j + step]
            local_var_map.append(np.var(patch))

    spatial_uniformity = np.std(local_var_map) if local_var_map else 0.0

    return [variance, skewness, kurtosis, spatial_uniformity]


def extract_edge_features(gray_img):
    """
    Analyzes edge characteristics across the image.
    Real images have physically-consistent edge distribution from optics.
    AI images often show unnaturally uniform or over-smoothed edges.
    """
    edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)

    total_pixels = edges.size
    edge_density = np.sum(edges > 0) / total_pixels

    h, w = edges.shape
    grid_size = 4
    cell_h = max(h // grid_size, 1)
    cell_w = max(w // grid_size, 1)

    cell_densities = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = edges[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            cell_densities.append(np.sum(cell > 0) / cell.size)

    return [
        edge_density,
        np.std(cell_densities),
        np.mean(cell_densities),
        np.max(cell_densities),
    ]


# ── single image pipeline ────────────────────────────────────────────────────


def extract_features_from_image(image_path: str) -> np.ndarray | None:
    """
    Loads a single image, preprocesses it, and returns a combined
    12-element feature vector (4 FFT + 4 noise + 4 edge).
    Returns None if the image cannot be loaded.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] Could not load: {image_path}")
        return None

    h, w = img.shape[:2]
    if h > 64 or w > 64:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    # else: use native resolution as-is
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fft_feats = extract_fft_features(gray)
    noise_feats = extract_noise_features(gray)
    edge_feats = extract_edge_features(gray)

    return np.array(fft_feats + noise_feats + edge_feats, dtype=np.float32)


# ── folder walker ────────────────────────────────────────────────────────────


def extract_features_from_folder(folder_path: str):
    """
    Recursively walks a folder, finds all images, extracts features in-place.

    Expected folder structure (typical for AI-detection datasets):
        folder/
            real/
                img1.jpg  img2.png ...
            fake/
                img1.jpg  img2.png ...

    But works with any arbitrary nesting — every subdir is counted separately.

    Returns:
        features   : np.ndarray of shape (N, 12)
        image_paths: list of Path objects, same order as features rows
        counts     : dict with per-subdir counts + total
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # ── count pass: walk tree and tally images per subdir ───────────────────
    subdir_counts = defaultdict(int)

    for root, dirs, files in os.walk(folder):
        root_path = Path(root)
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                # key is relative subdir path from the root folder
                rel = root_path.relative_to(folder)
                key = str(rel) if str(rel) != "." else "(root)"
                subdir_counts[key] += 1

    total_images = sum(subdir_counts.values())

    # ── print summary ────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Folder : {folder}")
    print(f"  Images found per subdirectory:")
    for subdir, count in sorted(subdir_counts.items()):
        print(f"    {subdir:<35} {count:>6} images")
    print(f"  {'TOTAL':<35} {total_images:>6} images")
    print(f"{'─' * 50}\n")

    # ── extraction pass ──────────────────────────────────────────────────────
    features = []
    image_paths = []
    failed = 0

    for root, dirs, files in os.walk(folder):
        # sort for deterministic ordering
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            print("Cac=ncun master")
            feature_vec = extract_features_from_image(fpath)
            if feature_vec is not None:
                features.append(feature_vec)
                image_paths.append(fpath)
                print("for each their own")
            else:
                failed += 1

    print(f" Extracted : {len(features)} / {total_images} images")
    if failed:
        print(f"  Failed : {failed} images (unreadable/corrupt)")
    print()

    counts = {
        "per_subdir": dict(subdir_counts),
        "total": total_images,
        "extracted": len(features),
        "failed": failed,
    }
    print("ayaya")

    return np.array(features, dtype=np.float32), image_paths, counts
