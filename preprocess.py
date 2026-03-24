from pathlib import Path
import numpy as np
from features import extract


def preprocess(path_cfg, feat_cfg):
    cache = Path(path_cfg.cache)
    features, labels, names = [], [], None

    for emotion_dir in sorted(Path(path_cfg.emotion).iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name
        samples = sorted(emotion_dir.glob("*.wav"))

        print(f"Processing {emotion}: {len(samples)} files")
        for sample in samples:
            augmented = extract(str(sample), feat_cfg, aug=True)
            if augmented is None:
                continue
            for feat_dict in augmented:
                names = names or list(feat_dict.keys())
                features.append(list(feat_dict.values()))
                labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)
    np.save(cache / path_cfg.feature, features)
    np.save(cache / path_cfg.label, labels)
    np.save(cache / path_cfg.name, np.array(names))

    print(f"Features: {features.shape}, Emotions: {np.unique(labels)}")
