from pathlib import Path
import numpy as np
from features import extract_features_augmented


def preprocess_data(emotions_dir='emotions'):
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)

    features_list, labels_list = [], []

    for emotion_folder in Path(emotions_dir).iterdir():
        if not emotion_folder.is_dir():
            continue
        emotion = emotion_folder.name
        audio_files = list(emotion_folder.glob('*.wav'))
        print(f"Processing {emotion}: {len(audio_files)} files")

        for audio_file in audio_files:
            try:
                features = extract_features_augmented(str(audio_file))
                if features is not None:
                    for feat in features:
                        features_list.append(feat)
                        labels_list.append(emotion)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    np.save(cache_dir / 'features_50.npy', features_array)
    np.save(cache_dir / 'labels_50.npy', labels_array)

    print(f"Features shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Emotions: {np.unique(labels_array)}")


if __name__ == '__main__':
    preprocess_data()