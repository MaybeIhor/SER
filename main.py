import random
import numpy as np
import torch
import preprocess
import train
import test


class PathConfig:
    cache = "cache"
    model = "model"
    test = "test"
    emotion = "emotions"
    feature = "features.npy"
    label = "labels.npy"
    name = "names.npy"
    scaler = "scaler.npy"
    encoder = "encoder.npy"
    focal = "focal.pth"
    log = "logs.txt"


class FeatureConfig:
    hop = 512
    sr = 22050
    mfcc_num = 40
    mfcc_stat = {0, 1, 2, 3, 5, 16, 17, 19, 22}
    mfcc_stat_delta = {0, 2, 6}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 12

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path_cfg = PathConfig()
    feature_cfg = FeatureConfig()

    preprocess.preprocess(path_cfg, feature_cfg)
    train.train(path_cfg, 200, device)
    test.test(path_cfg, feature_cfg, device)