import numpy as np
from pathlib import Path
import torch
from features import extract
from model import Model


def test(path_config, feature_config, device):
    checkpoint = torch.load(Path(path_config.model) / path_config.focal)
    model = Model(checkpoint['input_dim'], checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    encoder = np.load(Path(path_config.model) / path_config.encoder, allow_pickle=True).item()
    scaler = np.load(Path(path_config.model) / path_config.scaler, allow_pickle=True).item()

    for test_file in sorted(p for p in Path(path_config.test).iterdir() if p.suffix == '.wav'):
        features = extract(str(test_file), feature_config)
        features = np.array(list(features.values()))
        tensor = torch.FloatTensor(scaler.transform(features.reshape(1, -1))).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        idx = probs.argmax().item()
        scores = ', '.join(f"{encoder.classes_[i]}: {probs[i].item():.4f}" for i in range(len(encoder.classes_)))
        print(f"{test_file.name} - {encoder.classes_[idx]}      {scores}")