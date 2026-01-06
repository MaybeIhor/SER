import os
import numpy as np
from tensorflow import keras
import pickle

from features import extract_features


def predict_emotion(file_path, model, label_encoder, scaler):
    try:
        features = extract_features(file_path)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        emotion = label_encoder.inverse_transform([predicted_class])[0]

        all_probs = {label_encoder.inverse_transform([i])[0]: prediction[0][i]
                     for i in range(len(prediction[0]))}

        return emotion, confidence, all_probs
    except Exception as e:
        return None, None, str(e)


def main():

    print("Loading model...")
    model = keras.models.load_model('emotion_model.keras')

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    test_folder = 'test'

    audio_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.wav')])

    for file in audio_files:
        file_path = os.path.join(test_folder, file)
        emotion, confidence, all_probs = predict_emotion(file_path, model, label_encoder, scaler)

        if emotion:
            print(f"{file}: {emotion}")
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            print(f"  All: {', '.join([f'{e}: {p:.1%}' for e, p in sorted_probs])}")
            print()
        else:
            print(f"{file}: Error - {all_probs}")


if __name__ == "__main__":
    main()