from feature_extractor import extract_features_from_folder
from model import train

features, paths, counts = extract_features_from_folder("./data/CIFAKE")
model, scaler, acc = train(features, paths, save_dir="models")
