from feature_extractor import extract_features_from_folder

features, paths, counts = extract_features_from_folder("./data/CIFAKE")
# features.shape → (N, 12)
# paths[i]       → Path to the image that produced features[i]
# counts         → {'per_subdir': {...}, 'total': 4800, 'extracted': 4795, 'failed': 5}
