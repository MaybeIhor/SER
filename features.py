import numpy as np
import librosa


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    if len(audio) < sr * 3:
        audio = np.pad(audio, (0, sr * 3 - len(audio)))
    else:
        audio = audio[:sr * 3]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features_list = []

    for feat in [mfcc, mfcc_delta, mfcc_delta2]:
        features_list.append(np.mean(feat, axis=1))
        features_list.append(np.std(feat, axis=1))
        features_list.append(np.median(feat, axis=1))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features_list.append(np.mean(chroma, axis=1))
    features_list.append(np.std(chroma, axis=1))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features_list.append(np.mean(mel_db, axis=1))
    features_list.append(np.std(mel_db, axis=1))

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    features_list.append(np.mean(contrast, axis=1))
    features_list.append(np.std(contrast, axis=1))

    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    features_list.append(np.mean(tonnetz, axis=1))
    features_list.append(np.std(tonnetz, axis=1))

    zcr = librosa.feature.zero_crossing_rate(audio)
    features_list.append([np.mean(zcr), np.std(zcr)])

    rms = librosa.feature.rms(y=audio)
    features_list.append([np.mean(rms), np.std(rms), np.max(rms)])

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features_list.append([np.mean(spectral_centroid), np.std(spectral_centroid)])

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    features_list.append([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features_list.append([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if len(pitch_values) > 0:
        features_list.append([np.mean(pitch_values), np.std(pitch_values), np.median(pitch_values)])
    else:
        features_list.append([0, 0, 0])

    features = np.hstack([np.array(f).flatten() for f in features_list])

    return features