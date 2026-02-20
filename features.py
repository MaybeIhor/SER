import librosa
import numpy as np


def _compute_feature_stats(y, sr):
    hop_length = 512
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)

    stats = [np.mean(zcr), np.std(zcr), np.max(zcr), np.median(zcr),
             np.mean(rmse), np.std(rmse), np.max(rmse), np.median(rmse)]

    for i in range(40):
        if i in [20, 36]:
            continue
        stats += [np.mean(mfcc[i]), np.std(mfcc[i])]
        if i in [0, 1, 2, 3, 5, 16, 17, 19, 22]:
            stats += [np.min(mfcc[i]), np.max(mfcc[i]), np.median(mfcc[i])]

    for i in range(40):
        stats.append(np.std(mfcc_delta[i]))
        if i in [0, 2, 6]:
            stats += [np.min(mfcc_delta[i]), np.max(mfcc_delta[i])]

    rm, rs, rx = np.mean(rmse), np.std(rmse), np.max(rmse)
    zm, zs, zmd = np.mean(zcr), np.std(zcr), np.median(zcr)
    m0m, m0s, m0x = np.mean(mfcc[0]), np.std(mfcc[0]), np.max(mfcc[0])
    m2m, m2s, m2n = np.mean(mfcc[2]), np.std(mfcc[2]), np.min(mfcc[2])

    stats += [
        rx * zmd, rs * zs, rm / (zm + 1e-8),
        m0x * rx, m0m * rm, m0s * rs,
        m2s * m0s, m2n * m0m, m0m * m2m,
        rm * m0m * zm,
        np.sum(np.abs(np.diff(rmse))) / len(rmse),
        np.sum(np.abs(np.diff(zcr))) / len(zcr),
        np.sum(np.abs(np.diff(mfcc[0]))) / len(mfcc[0]),
    ]

    return np.array(stats)


def _load_and_normalize(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    if len(y) < sr * 0.5:
        return None, None
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / rms * 0.1
    return y, sr


def extract_features(file_path, sr=22050):
    y, sr = _load_and_normalize(file_path, sr)
    if y is None:
        return None
    return _compute_feature_stats(y, sr)


def augment_audio(y, sr):
    augmented = [y]
    if np.random.random() < 0.5:
        augmented.append(y + np.random.randn(len(y)) * 0.005)
    if np.random.random() < 0.5:
        augmented.append(librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2)))
    if np.random.random() < 0.5:
        n = np.random.randint(-2, 3)
        if n != 0:
            augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n))
    return augmented


def get_feature_names():
    names = ['zcr_mean', 'zcr_std', 'zcr_max', 'zcr_median',
             'rmse_mean', 'rmse_std', 'rmse_max', 'rmse_median']
    extended = {0, 1, 2, 3, 5, 16, 17, 19, 22}
    for i in range(40):
        if i in [20, 36]:
            continue
        names += [f'mfcc{i}_mean', f'mfcc{i}_std']
        if i in extended:
            names += [f'mfcc{i}_min', f'mfcc{i}_max', f'mfcc{i}_median']
    delta_extended = {0, 2, 6}
    for i in range(40):
        names.append(f'mfcc_delta{i}_std')
        if i in delta_extended:
            names += [f'mfcc_delta{i}_min', f'mfcc_delta{i}_max']
    names += [
        'rmse_max*zcr_median', 'rmse_std*zcr_std', 'rmse_mean/zcr_mean',
        'mfcc0_max*rmse_max', 'mfcc0_mean*rmse_mean', 'mfcc0_std*rmse_std',
        'mfcc2_std*mfcc0_std', 'mfcc2_min*mfcc0_mean', 'mfcc0_mean*mfcc2_mean',
        'rmse_mean*mfcc0_mean*zcr_mean',
        'rmse_flux', 'zcr_flux', 'mfcc0_flux',
    ]
    return names


def extract_features_augmented(file_path, sr=22050):
    y, sr = _load_and_normalize(file_path, sr)
    if y is None:
        return None
    return [_compute_feature_stats(s, sr) for s in augment_audio(y, sr)]