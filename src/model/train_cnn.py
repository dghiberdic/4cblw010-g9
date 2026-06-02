"""
train_cnn.py

1D CNN training script for FTIR functional-group prediction.

Current purpose:
- Define the CNN model architecture.
- Load processed X/y files once available.
- Train a multi-label functional-group classifier.

Expected data later:
X_spectra.npy : shape (n_samples, spectrum_length)
y_labels.npy  : shape (n_samples, n_functional_groups)
"""

from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    tf = None
    layers = None
    models = None


def find_data_files():
    """
    Looks for processed model-ready data files.
    These do not exist yet in the current repo, but this function defines
    the expected locations.
    """

    possible_x_paths = [
        Path("outputs/X_spectra.npy"),
        Path("outputs/X.npy"),
        Path("data/processed/X_spectra.npy"),
        Path("data/processed/X.npy"),
    ]

    possible_y_paths = [
        Path("outputs/y_labels.npy"),
        Path("outputs/y.npy"),
        Path("data/processed/y_labels.npy"),
        Path("data/processed/y.npy"),
    ]

    x_path = next((p for p in possible_x_paths if p.exists()), None)
    y_path = next((p for p in possible_y_paths if p.exists()), None)

    return x_path, y_path


def load_data():
    """
    Loads processed spectra and labels.

    X expected shape:
        (n_samples, spectrum_length)

    y expected shape:
        (n_samples, n_functional_groups)
    """

    x_path, y_path = find_data_files()

    if x_path is None or y_path is None:
        raise FileNotFoundError(
            "No processed X/y files found yet.\n"
            "Expected files like outputs/X_spectra.npy and outputs/y_labels.npy.\n"
            "Ask the data team to export model-ready spectra and labels."
        )

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Loaded X from: {x_path}")
    print(f"Loaded y from: {y_path}")
    print("X shape before CNN reshape:", X.shape)
    print("y shape:", y.shape)

    # CNN expects shape: (samples, length, channels)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    print("X shape after CNN reshape:", X.shape)

    return X, y


def build_ftir_cnn(input_length: int, num_labels: int):
    """
    Builds a 1D CNN for multi-label FTIR functional-group prediction.

    input_length:
        Number of points in one processed FTIR spectrum.

    num_labels:
        Number of functional groups predicted.
    """

    if tf is None:
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        )

    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),

        layers.Conv1D(32, kernel_size=7, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        # Sigmoid because this is multi-label:
        # multiple functional groups can be present at once.
        layers.Dense(num_labels, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    input_length = X_train.shape[1]
    num_labels = y_train.shape[1]

    model = build_ftir_cnn(input_length, num_labels)
    model.summary()

    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "outputs/models/ftir_cnn_best.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    print("CNN evaluation")
    print("Micro F1:", f1_score(y_test, y_pred, average="micro", zero_division=0))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
    print(classification_report(y_test, y_pred, zero_division=0))

    model.save("outputs/models/ftir_cnn_final.keras")


if __name__ == "__main__":
    main()
