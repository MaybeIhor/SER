import numpy as np
from librosa import effects, feature, load


def extract_features(y, sr, feature_cfg):
    zcr = feature.zero_crossing_rate(y, hop_length=feature_cfg.hop)[0]
    rmse = feature.rms(y=y, hop_length=feature_cfg.hop)[0]
    mfcc = feature.mfcc(y=y, sr=sr, n_mfcc=feature_cfg.mfcc_num, hop_length=feature_cfg.hop)
    mfcc_delta = feature.delta(mfcc)

    stats = dict()

    stats["zcr_mean"] = np.mean(zcr)
    stats["zcr_max"] = np.max(zcr)
    stats["zcr_std"] = np.std(zcr)
    stats["zcr_median"] = np.median(zcr)
    stats["rmse_mean"] = np.mean(rmse)
    stats["rmse_max"] = np.max(rmse)
    stats["rmse_std"] = np.std(rmse)
    stats["rmse_median"] = np.median(rmse)

    for i in range(feature_cfg.mfcc_num):
        stats[f"mfcc{i}_mean"] = np.mean(mfcc[i])
        stats[f"mfcc{i}_std"] = np.std(mfcc[i])
        stats[f"mfcc_delta{i}_std"] = np.std(mfcc_delta[i])
        if i in feature_cfg.mfcc_stat:
            stats[f"mfcc{i}_min"] = np.min(mfcc[i])
            stats[f"mfcc{i}_max"] = np.max(mfcc[i])
            stats[f"mfcc{i}_median"] = np.median(mfcc[i])
        if i in feature_cfg.mfcc_stat_delta:
            stats[f"mfcc_delta{i}_min"] = np.min(mfcc_delta[i])
            stats[f"mfcc_delta{i}_max"] = np.max(mfcc_delta[i])

    stats["rmse_max*zcr_median"] = stats["rmse_max"] * stats["zcr_median"]
    stats["rmse_std*zcr_std"] = stats["rmse_std"] * stats["zcr_std"]
    stats["rmse_mean/zcr_mean"] = stats["rmse_mean"] / (stats["zcr_mean"] + 1e-8)
    stats["mfcc0_max*rmse_max"] = stats["mfcc0_max"] * stats["rmse_max"]
    stats["mfcc0_mean*rmse_mean"] = stats["mfcc0_mean"] * stats["rmse_mean"]
    stats["mfcc0_std*rmse_std"] = stats["mfcc0_std"] * stats["rmse_std"]
    stats["mfcc2_std*mfcc0_std"] = stats["mfcc2_std"] * stats["mfcc0_std"]
    stats["mfcc2_min*mfcc0_mean"] = stats["mfcc2_min"] * stats["mfcc0_mean"]
    stats["mfcc0_mean*mfcc2_mean"] = stats["mfcc0_mean"] * stats["mfcc2_mean"]
    stats["rmse_mean*mfcc0_mean*zcr_mean"] = stats["rmse_mean"] * stats["mfcc0_mean"] * stats["zcr_mean"]
    stats["rmse_flux"] = np.sum(np.abs(np.diff(rmse))) / len(rmse)
    stats["zcr_flux"] = np.sum(np.abs(np.diff(zcr))) / len(zcr)
    stats["mfcc0_flux"] = np.sum(np.abs(np.diff(mfcc[0]))) / len(mfcc[0])

    return stats


def augment(y, sr):
    samples = []
    if np.random.random() < 0.5:
        samples.append(y + np.random.standard_normal(len(y)) * 0.005)
    if np.random.random() < 0.5:
        samples.append(effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2)))
    if np.random.random() < 0.5:
        n_steps = int(np.random.randint(-2, 3))
        if n_steps != 0:
            samples.append(effects.pitch_shift(y, sr=sr, n_steps=n_steps))
    return samples


def normalize(file_path, sr):
    y, sr = load(file_path, sr=sr)
    if len(y) < sr * 0.5:
        return None, None
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / rms * 0.1
    return y, sr


def extract(file_path, feature_cfg, aug=False):
    y, sr = normalize(file_path, feature_cfg.sr)
    if y is None:
        return None
    if aug:
        all_samples = [y] + augment(y, sr)
        return [extract_features(s, sr, feature_cfg) for s in all_samples]
    return extract_features(y, sr, feature_cfg)
