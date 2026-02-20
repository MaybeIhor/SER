import os
import pickle
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from features import get_feature_names
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

cache_dir = 'cache'
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

features_file = os.path.join(cache_dir, 'features_50.npy')
labels_file = os.path.join(cache_dir, 'labels_50.npy')

if not os.path.exists(features_file) or not os.path.exists(labels_file):
    print("Error: Preprocessed data not found. Run preprocess.py first.")
    exit(1)

X = np.load(features_file)
y = np.load(labels_file)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

class_counts = Counter(y_train)
class_weights = torch.FloatTensor(
    [len(y_train) / (len(class_counts) * class_counts[i]) for i in range(len(label_encoder.classes_))]
).to(device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


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


def make_loader(X, y, shuffle):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=128, shuffle=shuffle, num_workers=0)


train_loader = make_loader(X_train_scaled, y_train, shuffle=True)
test_loader = make_loader(X_test_scaled, y_test, shuffle=False)

model = CompactNet(X.shape[1], len(label_encoder.classes_)).to(device)
criterion = FocalLoss(gamma=1.2, weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-7)

print("\nTraining...")
best_acc = 0

for epoch in range(200):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch.to(device)), y_batch.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = model(X_batch.to(device)).argmax(1).cpu()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        acc = correct / total
        print(f"Epoch {epoch + 1}: Loss={train_loss / len(train_loader):.4f}, Val Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': model.state_dict(), 'input_dim': X.shape[1], 'num_classes': len(label_encoder.classes_)},
                       os.path.join(model_dir, 'focal_model_50.pth'))

print("\nFinal evaluation...")
model.load_state_dict(torch.load(os.path.join(model_dir, 'focal_model_50.pth'))['model_state_dict'])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        all_preds.extend(model(X_batch.to(device)).argmax(1).cpu().numpy())
        all_labels.extend(y_batch.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Best Val Acc: {best_acc:.4f}")
print(f"Final Test Acc: {accuracy:.4f}")

with open(os.path.join(model_dir, 'focal_artifacts_50.pkl'), 'wb') as f:
    pickle.dump({'label_encoder': label_encoder, 'scaler': scaler}, f)

with open('logs.txt', 'w') as log:
    log.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
    log.write(f"Final Test Accuracy: {accuracy:.4f}\n\n")

    log.write("Classification Report:\n")
    log.write(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    log.write("\n")

    log.write("Confusion Matrix:\n")
    log.write(" ".join(label_encoder.classes_) + "\n")
    cm = confusion_matrix(all_labels, all_preds)
    for row in cm:
        log.write(" ".join(map(str, row)) + "\n")
    log.write("\n")

    log.write("Feature Importance (permutation-based, mean accuracy drop):\n")
    model.eval()
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.LongTensor(y_test)
    with torch.no_grad():
        base_acc = (model(X_test_t).argmax(1).cpu() == y_test_t).float().mean().item()

    importances = []
    for i in range(X_test_scaled.shape[1]):
        X_perm = X_test_scaled.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_t = torch.FloatTensor(X_perm).to(device)
        with torch.no_grad():
            perm_acc = (model(perm_t).argmax(1).cpu() == y_test_t).float().mean().item()
        importances.append((i, base_acc - perm_acc))

    feature_names = get_feature_names()
    importances.sort(key=lambda x: x[1], reverse=True)
    log.write(f"Baseline accuracy: {base_acc:.4f}\n")
    for rank, (feat_idx, drop) in enumerate(importances, 1):
        log.write(f"{rank} {feature_names[feat_idx]} {drop:.4f}\n")

print("Model and logs saved.")