from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from model import Model, FocalLoss


def train(path_config, epochs, device):
    x = np.load(Path(path_config.cache) / path_config.feature)
    y = np.load(Path(path_config.cache) / path_config.label)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    train_loader = data_loader(x_train, y_train, shuffle=True)
    test_loader = data_loader(x_test, y_test, shuffle=False)

    counts = Counter(y_train)
    weights = torch.FloatTensor([len(y_train) / (num_classes * counts[i]) for i in range(num_classes)]).to(device)

    model = Model(x_train.shape[1], num_classes).to(device)
    criterion = FocalLoss(gamma=1.2, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)

    saved_model = Path(path_config.model) / path_config.focal
    best = 0
    for epoch in range(epochs):
        model.train()
        train_loss = sum(step(model, criterion, optimizer, x_b, y_b, device) for x_b, y_b in train_loader)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch + 1}: Loss={train_loss / len(train_loader):.4f}, Val Acc={acc:.4f}")
            if acc > best:
                best = acc
                torch.save({'model_state_dict': model.state_dict(), 'input_dim': x_train.shape[1], 'num_classes': num_classes}, saved_model)

    model.load_state_dict(torch.load(saved_model, weights_only=True)['model_state_dict'])

    predictions, labels = predict(model, test_loader, device)
    accuracy = accuracy_score(labels, predictions)
    print(f"Best Accuracy: {best:.4f}, Final Accuracy: {accuracy:.4f}")
    
    np.save(Path(path_config.model) / path_config.encoder, encoder)
    np.save(Path(path_config.model) / path_config.scaler, scaler)

    imp = importance(model, x_test, y_test, device)
    ranking = np.argsort(imp)[::-1]

    with open(path_config.log, 'w') as log:
        names = np.load(Path(path_config.cache) / path_config.name).tolist()
        log.write(f"Best Accuracy: {best:.4f}\nFinal Accuracy: {accuracy:.4f}\n\n")
        log.write("Classification Report:\n")
        log.write(classification_report(labels, predictions, target_names=encoder.classes_))
        log.write("\nConfusion Matrix:\n")
        log.write(" ".join(encoder.classes_) + "\n")
        log.writelines(" ".join(map(str, row)) + "\n" for row in confusion_matrix(labels, predictions))
        log.write("\nFeatures Importance:\n")
        log.writelines(f"{rank} {names[idx]} {imp[idx]:.4f}\n" for rank, idx in enumerate(ranking, 1))


def step(model, criterion, optimizer, x, y, device):
    optimizer.zero_grad()
    loss = criterion(model(x.to(device)), y.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            predictions.extend(model(x_b.to(device)).argmax(1).cpu().numpy())
            labels.extend(y_b.numpy())
    return predictions, labels


def evaluate(model, loader, device):
    predictions, labels = predict(model, loader, device)
    return accuracy_score(labels, predictions)


def importance(model, x, y, device):
    rand = np.random.default_rng(42)
    y_ten = torch.LongTensor(y)
    with torch.no_grad():
        acc = (model(torch.FloatTensor(x).to(device)).argmax(1).cpu() == y_ten).float().mean().item()
    imp = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        x_imp = x.copy()
        x_imp[:, i] = rand.permutation(x_imp[:, i])
        with torch.no_grad():
            acc_imp = (model(torch.FloatTensor(x_imp).to(device)).argmax(1).cpu() == y_ten).float().mean().item()
        imp[i] = acc - acc_imp
    return imp


def data_loader(x, y, shuffle):
    return DataLoader(TensorDataset(torch.FloatTensor(x), torch.LongTensor(y)), batch_size=128, shuffle=shuffle, num_workers=0)
