import os, time, pickle, joblib, json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, log_loss, roc_auc_score,
    matthews_corrcoef, confusion_matrix, cohen_kappa_score
)

# --------------------------- Config ---------------------------
BATCH_SIZE   = 512
EPOCHS       = 50
PATIENCE     = 10
RANDOM_SEED  = 42

TRAIN_PATH = 'balanced_gan_synthetic_data.csv'
TEST_PATH  = 'test_dataset_adaptive.csv'

# Output dirs
os.makedirs("models/h5", exist_ok=True)
os.makedirs("models/tflite", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


np.random.seed(RANDOM_SEED); tf.random.set_seed(RANDOM_SEED)
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# --------------------------- Data ---------------------------
def load_and_preprocess():
    # 1) Load CSVs
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # 2) Target extraction
    if "fault" not in train_df.columns or "fault" not in test_df.columns:
        raise ValueError("Both train and test must contain a 'fault' column as the target.")
    y_train = train_df["fault"].to_numpy()
    y_test  = test_df["fault"].to_numpy()

    # 3) Remove target and helper cols if present
    drop_cols = ["fault", "data_source"]
    X_train_df = train_df.drop(columns=drop_cols, errors="ignore")
    X_test_df  = test_df.drop(columns=drop_cols, errors="ignore")

    # 4) Keep numeric features only (avoid columns like 'type', ids, strings, etc.)
    num_train = X_train_df.select_dtypes(include=[np.number])
    num_test  = X_test_df.select_dtypes(include=[np.number])

    # 5) Align to common numeric columns with a consistent, sorted order
    common = sorted(set(num_train.columns) & set(num_test.columns))
    if len(common) == 0:
        raise ValueError(
            "No common numeric features between train and test after cleaning. "
            "Check your CSVs for mismatched column names."
        )

    X_train = num_train[common].to_numpy(dtype=np.float32)
    X_test  = num_test[common].to_numpy(dtype=np.float32)

    # 6) Scale with the same feature set and order
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 7) Persist scaler + features for reproducibility
    joblib.dump({"scaler": scaler, "features": common}, "results/scaler_and_features.joblib")

    # 8) Train/val split
    split = int(0.8 * len(X_train))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X_train))

    data = {
        "X_train": X_train[idx[:split]],
        "y_train": y_train[idx[:split]],
        "X_val":   X_train[idx[split:]],
        "y_val":   y_train[idx[split:]],
        "X_test":  X_test,
        "y_test":  y_test,
        "features": common,
    }

    print(f"[Data] Using {len(common)} aligned numeric features:")
    print(common)
    print(f"[Data] Shapes -> X_train: {data['X_train'].shape}, X_val: {data['X_val'].shape}, X_test: {data['X_test'].shape}")
    return data

def create_ds(X, y, train=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train: ds = ds.shuffle(10000).cache()
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --------------------------- Models ---------------------------
def build_cnn(input_dim, n_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="CNN")

def build_lstm(input_dim, n_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="LSTM")

def build_rnn(input_dim, n_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(x)
    x = tf.keras.layers.SimpleRNN(32)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="RNN")

# --------------------------- Train/Eval ---------------------------
def train_and_eval(model_fn, data, name):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    n_classes = int(len(np.unique(y_train)))
    model = model_fn(X_train.shape[1], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-5, verbose=1)
    ]

    start = time.time()
    history = model.fit(
        create_ds(X_train, y_train, True),
        validation_data=create_ds(X_val, y_val, False),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start

    # Save .h5
    h5_path = f"models/h5/{name}.h5"
    model.save(h5_path)

    # Save .tflite (best-effort)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        with open(f"models/tflite/{name}.tflite", "wb") as f:
            f.write(tflite_model)
    except Exception as e:
        print(f"[TFLite] Conversion failed for {name}: {e}")

    # Evaluate
    y_proba = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    metrics = {
        "accuracy":       accuracy_score(y_test, y_pred),
        "balanced_acc":   balanced_accuracy_score(y_test, y_pred),
        "precision":      precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":         recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":             f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "log_loss":       log_loss(y_test, y_proba),
        "cohen_kappa":    cohen_kappa_score(y_test, y_pred),
        "mcc":            matthews_corrcoef(y_test, y_pred),
        "train_time":     float(train_time),
        "params":         int(model.count_params())
    }

    # Robust ROC-AUC (may fail if a class is missing in y_test)
    try:
        y_onehot = tf.keras.utils.to_categorical(y_test, y_proba.shape[1])
        metrics["roc_auc"] = roc_auc_score(y_onehot, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        metrics["roc_auc"] = np.nan

    # Plots: training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title(f"{name} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title(f"{name} Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout(); plt.savefig(f"plots/{name}_training.png"); plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix\nAccuracy: {metrics['accuracy']:.4f}")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(f"plots/{name}_cm.png"); plt.close()

    return metrics

# --------------------------- Main ---------------------------
def main():
    data = load_and_preprocess()

    results = {}
    for builder, name in [(build_cnn, "CNN"), (build_lstm, "LSTM"), (build_rnn, "RNN")]:
        print(f"\n=== Training {name} ===")
        results[name] = train_and_eval(builder, data, name)

   
    with open("results/all_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Comparison dataframe
    df = pd.DataFrame(results).T
    df = df[["accuracy", "f1", "precision", "recall", "balanced_acc", "roc_auc", "log_loss", "train_time", "params"]]
    df.to_csv("results/summary.csv", index=True)

    # Comparison bar plot for key scores
    metrics_plot = ["accuracy", "f1", "precision", "recall", "balanced_acc"]
    plt.figure(figsize=(12,6))
    df[metrics_plot].plot(kind="bar")
    plt.title("Model Comparison (Higher is Better)")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("plots/comparison_metrics.png"); plt.close()

    print("\n=== Summary ===")
    print(df)

if __name__=="__main__":
    main()
