from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_labels(image_paths: list) -> np.ndarray:
    """
    Derives labels from folder names in the image path.
    0 = REAL, 1 = FAKE
    """
    labels = []
    for path in image_paths:
        parts = [p.upper() for p in Path(path).parts]
        if "FAKE" in parts:
            labels.append(1)
        elif "REAL" in parts:
            labels.append(0)
        else:
            raise ValueError(f"Could not determine label from path: {path}")
    return np.array(labels, dtype=np.int32)


def train(features: np.ndarray, image_paths: list, save_dir: str = "models"):
    """
    Trains an SVM on extracted features and saves the model + scaler.
    """
    labels = build_labels(image_paths)

    # -- scale features --
    # SVM is sensitive to feature scale — FFT values might be in hundreds
    # while noise variance might be 0.003. Without scaling, large-valued
    # features dominate and the model learns the wrong boundary.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # -- train/test split --
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,  # keeps REAL/FAKE ratio equal in both splits
    )

    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(
        f"  Label distribution — REAL: {(labels == 0).sum()} | FAKE: {(labels == 1).sum()}\n"
    )

    # -- train SVM --
    # RBF kernel handles non-linear boundaries between AI and real features.
    # probability=True lets us return a confidence score, not just a class.
    print("  Training SVM...")
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -- save --
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_path / "svm_model.pkl")
    joblib.dump(scaler, save_path / "scaler.pkl")

    print(f"\n  Model  saved → {save_path / 'svm_model.pkl'}")
    print(f"  Scaler saved → {save_path / 'scaler.pkl'}")

    return model, scaler, acc
