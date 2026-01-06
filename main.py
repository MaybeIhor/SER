import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
import pickle

from features import extract_features

def augment_data(x, y, augmentation_factor=2):
    x_aug = list(x)
    y_aug = list(y)

    for _ in range(augmentation_factor):
        for features, label in zip(x, y):
            noise = np.random.normal(0, 0.003, features.shape)
            x_aug.append(features + noise)
            y_aug.append(label)

    return np.array(x_aug), np.array(y_aug)


def load_data(folders):
    features = []
    labels = []

    for folder in folders:
        if not os.path.exists(folder):
            continue

        emotion = os.path.basename(folder)

        for file in os.listdir(folder):
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(folder, file)
                try:
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(features), np.array(labels)


def create_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    folders = ['angry', 'fear', 'happy']

    print("Loading data...")
    x, y = load_data(folders)

    print(f"Loaded {len(x)} samples")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training samples (before augmentation): {len(x_train)}, Test samples: {len(x_test)}")

    print("Augmenting training data...")
    x_train_aug, y_train_aug = augment_data(x_train, y_train, augmentation_factor=2)
    print(f"After augmentation: {len(x_train_aug)} training samples")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_aug)
    x_test_scaled = scaler.transform(x_test)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_aug),
        y=y_train_aug
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    model = create_model(x_train_scaled.shape[1], len(le.classes_))

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.000001
    )

    print("Training model...")
    model.fit(
        x_train_scaled, y_train_aug,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test_scaled, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    model.save('emotion_model.keras')

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()