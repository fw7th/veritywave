import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from feature_extractor import (
    extract_edge_features,
    extract_features_from_image,
    extract_fft_features,
    extract_noise_features,
)

# ── video handler ────────────────────────────────────────────────────────────


def extract_frames(video_path: str, sample_every: int = 10) -> list:
    """
    Samples every Nth frame from a video.
    Returns list of grayscale frames as numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            # resize and convert to gray in place
            h, w = frame.shape[:2]
            if h > 64 or w > 64:
                frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        frame_idx += 1

    cap.release()
    print(f"  Sampled {len(frames)} frames from {frame_idx} total frames")
    return frames


def predict_video(video_path: str, model, scaler, sample_every: int = 10) -> dict:
    """
    Predicts whether a video is AI-generated or real.
    Extracts features per sampled frame, aggregates via mean probability.
    """
    frames = extract_frames(video_path, sample_every)

    if not frames:
        raise ValueError("No frames could be extracted from video.")

    frame_features = []
    for gray in frames:
        fft_feats = extract_fft_features(gray)
        noise_feats = extract_noise_features(gray)
        edge_feats = extract_edge_features(gray)
        vec = np.array(fft_feats + noise_feats + edge_feats, dtype=np.float32)
        frame_features.append(vec)

    frame_features = np.array(frame_features)
    frame_features_scaled = scaler.transform(frame_features)

    # probability per frame → mean across all frames
    probs = model.predict_proba(frame_features_scaled)  # shape (N, 2)
    mean_probs = probs.mean(axis=0)  # [p_real, p_fake]

    p_fake = float(mean_probs[1])
    p_real = float(mean_probs[0])
    prediction = "AI" if p_fake > p_real else "Real"
    confidence = round(max(p_fake, p_real), 4)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "frames_analyzed": len(frames),
        "p_real": round(p_real, 4),
        "p_fake": round(p_fake, 4),
    }


# ── main predictor ───────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def predict(input_path: str, model_dir: str = "models") -> dict:
    """
    Main entry point. Accepts an image or video path.
    Loads model + scaler, runs prediction, returns JSON-serializable dict.

    Output:
        {
            "prediction" : "AI" | "Real",
            "confidence" : 0.84,
            "input"      : "path/to/file.jpg",
            "type"       : "image" | "video"
        }
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # ── load model + scaler ──────────────────────────────────────────────────
    model_path = Path(model_dir) / "svm_model.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Model files not found in: {model_dir}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    suffix = path.suffix.lower()

    # ── image prediction ─────────────────────────────────────────────────────
    if suffix in IMAGE_EXTENSIONS:
        features = extract_features_from_image(str(path))
        if features is None:
            raise ValueError(f"Could not extract features from: {input_path}")

        features_scaled = scaler.transform(features.reshape(1, -1))
        probs = model.predict_proba(features_scaled)[0]  # [p_real, p_fake]
        p_real, p_fake = float(probs[0]), float(probs[1])
        prediction = "AI" if p_fake > p_real else "Real"
        confidence = round(max(p_fake, p_real), 4)

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "input": str(path),
            "type": "image",
            "p_real": round(p_real, 4),
            "p_fake": round(p_fake, 4),
        }

    # ── video prediction ─────────────────────────────────────────────────────
    elif suffix in VIDEO_EXTENSIONS:
        result = predict_video(str(path), model, scaler)
        result["input"] = str(path)
        result["type"] = "video"

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # ── print + return ───────────────────────────────────────────────────────
    print(json.dumps(result, indent=2))
    return result
