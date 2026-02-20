import os
import pickle
import torch
import torch.nn as nn
from features import extract_features

model_dir = 'model'
test_dir = 'test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CompactNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


checkpoint = torch.load(os.path.join(model_dir, 'focal_model_0.pth'))
model = CompactNet(checkpoint['input_dim'], checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open(os.path.join(model_dir, 'focal_artifacts_0.pkl'), 'rb') as f:
    artifacts = pickle.load(f)
    label_encoder = artifacts['label_encoder']
    scaler = artifacts['scaler']

for test_file in sorted(f for f in os.listdir(test_dir) if f.endswith('.wav')):
    try:
        features = extract_features(os.path.join(test_dir, test_file))
        tensor = torch.FloatTensor(scaler.transform(features.reshape(1, -1))).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
        idx = probs.argmax(1).item()
        print(f"{test_file}: {label_encoder.classes_[idx]} (conf: {probs[0, idx].item():.4f})")
    except Exception as e:
        print(f"Error processing {test_file}: {e}")
