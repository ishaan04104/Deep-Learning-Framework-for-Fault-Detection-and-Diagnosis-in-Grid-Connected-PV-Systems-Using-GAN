# ============================== train_with_feature_selection.py ==============================
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             precision_score, recall_score, log_loss,
                             roc_auc_score, matthews_corrcoef, confusion_matrix,
                             cohen_kappa_score)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc

# --------------------------- Config ---------------------------
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15
RANDOM_SEED = 42

FEATURE_SETS = {
    'boruta': ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc'],
    'lasso':  ['Vf', 'Vpv', 'Time', 'Vdc', 'Ipv', 'If', 'Vabc', 'ic', 'va', 'ib'],
    'rfe':    ['Ipv', 'Vpv', 'Vdc', 'ib', 'vb', 'vc', 'Iabc', 'If', 'Vabc', 'Vf'],
    'dt':     ['Vpv', 'Vabc', 'Ipv', 'Iabc', 'If', 'Time', 'Vdc', 'Vf', 'ic', 'ib'],
    'xgb':    ['Vpv', 'Ipv', 'If', 'Vdc', 'Vabc', 'Iabc', 'Time', 'Vf', 'ic', 'ib']
}

# --------------------------- Feature Selection Config ---------------------------

FEATURE_SELECTION_MODE = "majority"  
VOTE_THRESHOLD = 3                   
TOPK_PER_METHOD = 6                   

def _select_features_from_sets(feature_sets: dict,
                               mode: str = "majority",
                               vote_threshold: int = 3,
                               topk_each: int = 6):
    """
    Returns: (selected_features: List[str], tag: str)
    'tag' is appended to all filenames (scaler, models, plots, results).
    """
    sets = {k: list(v) for k, v in feature_sets.items()}
    all_feats = sorted({f for v in sets.values() for f in v})

    if mode == "all":
        return all_feats, "fs-all"

    if mode == "union":
        return all_feats, "fs-union"

    if mode == "intersection":
        inter = set(all_feats)
        for lst in sets.values():
            inter &= set(lst)
        return sorted(inter), "fs-intersection"

    if mode == "topk_each":
        chosen = []
        for _, lst in sets.items():
            chosen.extend(lst[:min(topk_each, len(lst))])
        # unique while preserving insertion order
        chosen = list(dict.fromkeys(chosen))
        return sorted(chosen), f"fs-topk{topk_each}"

    if mode == "majority":
        from collections import Counter
        c = Counter()
        for lst in sets.values():
            c.update(lst)
        maj = sorted([f for f, cnt in c.items() if cnt >= vote_threshold])
        return maj, f"fs-majority{vote_threshold}"

    return all_feats, "fs-union"

TRAIN_DATASET_PATH = '/balanced_gan_synthetic_data.csv'
TEST_DATASET_PATH  = '/test_dataset_adaptive.csv'


for dir_path in ['models/h5', 'models/tflite', 'scalers', 'results', 'plots']:
    os.makedirs(dir_path, exist_ok=True)

tf.config.optimizer.set_jit(False)  


# MODEL ARCHITECTURES 

def create_qcnn(input_shape, num_classes):
    """Lightweight Quantized CNN for efficient inference"""
    input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def attention_layer(inputs):
    """Keras 3 compatible simple attention"""
    time_steps = int(inputs.shape[1])
    a = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    a = tf.keras.layers.Reshape((time_steps,))(a)
    a = tf.keras.layers.Softmax()(a)
    a = tf.keras.layers.Reshape((time_steps, 1))(a)
    out = tf.keras.layers.Multiply()([inputs, a])
    out = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(out)
    return out

def create_lstm(input_shape, num_classes):
    """Enhanced TinyLSTM with bidirectional LSTM + conv + LSTM + attention"""
    input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.LSTM(16, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = attention_layer(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_mlp(input_shape, num_classes):
    """Carbon-efficient MLP"""
    input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_fbnet(input_shape, num_classes):
    """Lightweight FBNet-like MLP with residuals"""
    input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    skip = x
    x = tf.keras.layers.Dense(128, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.ReLU()(x)

    skip = x
    x = tf.keras.layers.Dense(128, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# DATA HANDLING 


def load_and_preprocess_data():
    """
    Load CSVs, select features by configured strategy, scale, split.
    Artifacts are tagged with fs_tag (e.g., fs-majority3, fs-intersection, etc.)
    """
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    test_df  = pd.read_csv(TEST_DATASET_PATH)

    # meta cols never used as features
    drop_cols = ['fault', 'data_source', 'type']
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df['fault'].values
    X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test  = test_df['fault'].values

    # choose feature subset
    selected_features, fs_tag = _select_features_from_sets(
        FEATURE_SETS,
        mode=FEATURE_SELECTION_MODE,
        vote_threshold=VOTE_THRESHOLD,
        topk_each=TOPK_PER_METHOD
    )
    # safety: ensure present in both
    missing_train = [c for c in selected_features if c not in X_train.columns]
    missing_test  = [c for c in selected_features if c not in X_test.columns]
    if missing_train or missing_test:
        raise ValueError(f"Selected features missing in data.\n"
                         f"Missing in train: {missing_train}\nMissing in test: {missing_test}")

    print(f"Feature selection mode: {FEATURE_SELECTION_MODE} | Tag: {fs_tag}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    # align
    X_train = X_train[selected_features]
    X_test  = X_test[selected_features]

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled  = scaler.transform(X_test.values)

    scaler_path = f"scalers/{fs_tag}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # split
    train_size = int(0.8 * len(X_train_scaled))
    indices = np.random.permutation(len(X_train_scaled))

    data = {
        'X_train': X_train_scaled[indices[:train_size]].astype('float32'),
        'y_train': y_train[indices[:train_size]],
        'X_val':   X_train_scaled[indices[train_size:]].astype('float32'),
        'y_val':   y_train[indices[train_size:]],
        'X_test':  X_test_scaled.astype('float32'),
        'y_test':  y_test,
        'features': selected_features,
        'fs_tag': fs_tag
    }
    return data


# TRAINING & EVALUATION 

def create_tf_dataset(X, y, batch_size, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        ds = ds.shuffle(10000).cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train_model(model_name, create_model_fn, data_dict):
    start_time = time.time()

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val     = data_dict['X_val'], data_dict['y_val']

    train_dataset = create_tf_dataset(X_train, y_train, BATCH_SIZE, True)
    val_dataset   = create_tf_dataset(X_val,   y_val,   BATCH_SIZE, False)

    input_shape = (X_train.shape[1],)
    n_classes   = len(np.unique(y_train))

    model = create_model_fn(input_shape, n_classes)

    if model_name == 'TinyLSTM':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    return model, history, training_time

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    n_classes = y_pred_proba.shape[1]
    y_test_onehot = tf.keras.utils.to_categorical(y_test, n_classes)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'log_loss': log_loss(y_test, y_pred_proba),
        'roc_auc': roc_auc_score(y_test_onehot, y_pred_proba, multi_class='ovr', average='weighted'),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    return metrics, y_pred


def analyze_model_efficiency(model, model_name, fs_tag):
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    tmp_path = f"models/h5/{fs_tag}_temp_{model_name}.h5"
    model.save(tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024*1024)
    os.remove(tmp_path)

    print(f"\n{model_name} Efficiency:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,}")
    print(f"  H5 size: {size_mb:.2f} MB")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'size_mb': size_mb
    }

def convert_to_tflite(model, model_name, fs_tag):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        out = f"models/tflite/{fs_tag}_union_{model_name}.tflite"
        with open(out, "wb") as f:
            f.write(tflite_model)
        size_kb = os.path.getsize(out) / 1024.0
        print(f"TFLite saved: {out} ({size_kb:.1f} KB)")
        return size_kb
    except Exception as e:
        print(f"Error converting to TFLite: {e}")
        return 0.0

def plot_training_history(history, model_name, fs_tag):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{model_name} Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'{model_name} Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{fs_tag}_union_{model_name}_training.png')
    plt.close()


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                print("Could not set memory growth")

    data = load_and_preprocess_data()
    fs_tag = data['fs_tag']

    # Models to train (unchanged set)
    models = {
        'QCNN': create_qcnn,
        'TinyLSTM': create_lstm,
        'CarbonMLP': create_mlp,
        'FBNet': create_fbnet
    }

    results = {}
    efficiency_metrics = {}

    for model_name, create_fn in models.items():
        results[model_name] = {}
        print(f"\n=== Training {model_name} ===")

        model, history, train_time = train_model(model_name, create_fn, data)

        efficiency_metrics[model_name] = analyze_model_efficiency(model, model_name, fs_tag)
        efficiency_metrics[model_name]['training_time'] = train_time

        plot_training_history(history, model_name, fs_tag)

        # Save H5 with tag
        h5_path = f"models/h5/{fs_tag}_union_{model_name}.h5"
        model.save(h5_path)
        print(f"Model saved: {h5_path}")

        # TFLite with tag
        tflite_kb = convert_to_tflite(model, model_name, fs_tag)
        efficiency_metrics[model_name]['tflite_size_kb'] = tflite_kb

        # Evaluate
        metrics, y_pred = evaluate_model(model, data['X_test'], data['y_test'])
        results[model_name] = {'metrics': metrics, 'train_time': train_time}

        print(f"\nResults for {model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Confusion matrix with tag
        cm = confusion_matrix(data['y_test'], y_pred)
        plt.figure(figsize=(9,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        # careful with f-string quoting inside dict indexing
        plt.title(f"{model_name} Confusion Matrix\nAccuracy: {metrics['accuracy']:.4f}")
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(f'plots/{fs_tag}_union_{model_name}_cm.png')
        plt.close()

        tf.keras.backend.clear_session()
        gc.collect()

    # Persist results/efficiency with tag
    import pickle
    with open(f'results/{fs_tag}_union_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(f'results/{fs_tag}_efficiency_metrics.pkl', 'wb') as f:
        pickle.dump(efficiency_metrics, f)

    # Summary
    print("\n=== Model Performance and Efficiency Summary ===")
    print(f"{'Model':<10} {'Accuracy':<10} {'F1':<10} {'Params':<12} {'Size (MB)':<10} {'Time (s)':<10}")
    print("-" * 60)
    for model_name in models.keys():
        m = results[model_name]['metrics']
        e = efficiency_metrics[model_name]
        print(f"{model_name:<10} {m['accuracy']:<10.4f} {m['f1']:<10.4f} "
              f"{e['total_params']:<12,} {e['size_mb']:<10.2f} {e['training_time']:<10.2f}")

    print("\nTraining complete! All models and results saved.")

if __name__ == "__main__":
    main()

