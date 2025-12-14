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

# Configuration
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15
RANDOM_SEED = 42
FEATURE_SETS = {
    'boruta': ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc'],
    'lasso': ['Vf', 'Vpv', 'Time', 'Vdc', 'Ipv', 'If', 'Vabc', 'ic', 'va', 'ib'],
    'rfe': ['Ipv', 'Vpv', 'Vdc', 'ib', 'vb', 'vc', 'Iabc', 'If', 'Vabc', 'Vf'],
    'dt': ['Vpv', 'Vabc', 'Ipv', 'Iabc', 'If', 'Time', 'Vdc', 'Vf', 'ic', 'ib'],
    'xgb': ['Vpv', 'Ipv', 'If', 'Vdc', 'Vabc', 'Iabc', 'Time', 'Vf', 'ic', 'ib']
}

# Paths to datasets
TRAIN_DATASET_PATH = 'balanced_gan_synthetic_data.csv'
TEST_DATASET_PATH = 'test_dataset_adaptive.csv'

# Create directories
for dir_path in ['models/h5', 'models/tflite', 'scalers', 'results', 'plots']:
    os.makedirs(dir_path, exist_ok=True)

# Enable XLA compilation for performance
tf.config.optimizer.set_jit(True)

# ---------------------------
# MODEL ARCHITECTURES
# ---------------------------

def create_qcnn(input_shape, num_classes):
    """
    Lightweight Quantized CNN for efficient inference
    """
    if isinstance(input_shape, tuple):
        input_dim = input_shape[0]
    else:
        input_dim = input_shape
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        # First convolutional block - efficient architecture
        tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        # Second convolutional block
        tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        # Feature extraction
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# --------------------------- Serializable Attention (replaces attention_layer) ---------------------------
class TemporalAttention(tf.keras.layers.Layer):
    """
    Serializable temporal attention for 3D inputs (batch, time, features).
    Returns a context vector (batch, features) via a softmax-weighted sum over time.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, time_steps, features)
        self.time_steps = int(input_shape[1])
        self.features   = int(input_shape[2])
        self.W = self.add_weight(
            shape=(self.features, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attn_W'
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name='attn_b'
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, F)
        e = tf.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b)  # (B, T, 1)
        a = tf.nn.softmax(tf.squeeze(e, axis=-1), axis=1)               # (B, T)
        a = tf.expand_dims(a, axis=-1)                                   # (B, T, 1)
        context = tf.reduce_sum(a * x, axis=1)                           # (B, F)
        return context

# --------------------------- TinyLSTM (replaces original create_lstm) ---------------------------
def create_lstm(input_shape, num_classes):
    """
    Enhanced TinyLSTM with bidirectional architecture and serializable attention
    (no Lambda).
    """
    if isinstance(input_shape, tuple):
        input_dim = input_shape[0]
    else:
        input_dim = input_shape
    
    # Create model using Functional API for attention mechanism
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    
    # Bidirectional first LSTM layer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add convolutional layer for hybrid approach
    x = tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu')(x)
    
    # Second LSTM layer 
    x = tf.keras.layers.LSTM(16, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Serializable attention mechanism
    x = TemporalAttention()(x)
    
    # Final dense layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_mlp(input_shape, num_classes):
    """
    Carbon-efficient MLP with improved accuracy
    """
    if isinstance(input_shape, tuple):
        input_dim = input_shape[0]
    else:
        input_dim = input_shape
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        # First dense layer
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_initializer='glorot_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        # Second dense layer 
        tf.keras.layers.Dense(32, activation='relu',
                             kernel_initializer='glorot_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_fbnet(input_shape, num_classes):
    """
    Lightweight FBNet implementation
    """
    if isinstance(input_shape, tuple):
        input_dim = input_shape[0]
    else:
        input_dim = input_shape
    
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Initial projection
    x = tf.keras.layers.Dense(64, activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # First inverted bottleneck block
    x_skip1 = x
    x = tf.keras.layers.Dense(128, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip1])
    x = tf.keras.layers.ReLU()(x)
    
    # Second inverted bottleneck block
    x_skip2 = x
    x = tf.keras.layers.Dense(128, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip2])
    x = tf.keras.layers.ReLU()(x)
    
    # Output layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ---------------------------
# DATA HANDLING FUNCTIONS
# ---------------------------

def load_and_preprocess_data():
    """Load and preprocess data with union features only"""
    print("Loading data...")
    
    # Load data
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    test_df = pd.read_csv(TEST_DATASET_PATH)
    
    # Handle data_source column if present
    if 'data_source' in train_df.columns:
        print(f"Data source distribution: {train_df['data_source'].value_counts()}")
        X_train = train_df.drop(['fault', 'data_source'], axis=1)
    else:
        X_train = train_df.drop('fault', axis=1)
    
    y_train = train_df['fault'].values
    X_test = test_df.drop('fault', axis=1)
    y_test = test_df['fault'].values
    
    # Calculate union of features
    union_features = set()
    for features in FEATURE_SETS.values():
        union_features.update(features)
    union_features = sorted(list(union_features))
    
    print(f"Union features ({len(union_features)}): {union_features}")
    
    # Select and scale features
    X_train_selected = X_train[union_features].values
    X_test_selected = X_test[union_features].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Save scaler
    joblib.dump(scaler, "scalers/union_scaler.pkl")
    
    # Split train into train/validation
    train_size = int(0.8 * len(X_train_scaled))
    indices = np.random.permutation(len(X_train_scaled))
    
    # Store data
    scaled_data = {
        'X_train': X_train_scaled[indices[:train_size]].astype('float32'),
        'y_train': y_train[indices[:train_size]],
        'X_val': X_train_scaled[indices[train_size:]].astype('float32'),
        'y_val': y_train[indices[train_size:]],
        'X_test': X_test_scaled.astype('float32'),
        'y_test': y_test,
        'features': union_features
    }
    
    return scaled_data

# ---------------------------
# TRAINING & EVALUATION
# ---------------------------

def create_tf_dataset(X, y, batch_size, is_training=True):
    """Create optimized TF Dataset for faster data loading"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model(model_name, create_model_fn, data_dict):
    """Train model with optimized learning rate schedule"""
    start_time = time.time()
    
    # Get basic info
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    # Create datasets
    train_dataset = create_tf_dataset(X_train, y_train, BATCH_SIZE)
    val_dataset = create_tf_dataset(X_val, y_val, BATCH_SIZE, is_training=False)
    
    # Get dimensions
    input_shape = (X_train.shape[1],)
    n_classes = len(np.unique(y_train))
    
    # Create model
    model = create_model_fn(input_shape, n_classes)
    
    # Use Keras 3 compatible optimizers
    if model_name == 'TinyLSTM':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better convergence
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
    
    # Train model
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
    """Comprehensive evaluation with all requested metrics"""
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # One-hot encode for multi-class ROC AUC
    n_classes = y_pred_proba.shape[1]
    y_test_onehot = tf.keras.utils.to_categorical(y_test, n_classes)
    
    # Calculate all requested metrics
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

# ---------------------------
# TFLITE CONVERSION
# ---------------------------

def convert_to_tflite(model, model_name, data_dict):
    """Convert model to TFLite with weight quantization"""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        tflite_filename = f"models/tflite/union_{model_name}.tflite"
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)
        tflite_size = os.path.getsize(tflite_filename) / 1024  # Size in KB
        print(f"TFLite model saved ({tflite_size:.1f} KB): {tflite_filename}")
        return tflite_size
    except Exception as e:
        print(f"Error in TFLite conversion: {e}")
        return 0

# ---------------------------
# VISUALIZATION & ANALYSIS
# ---------------------------

def plot_training_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/union_{model_name}_training.png')
    plt.close()

def analyze_model_efficiency(model, model_name):
    """Analyze model size and parameter count"""
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n{model_name} Efficiency Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # Calculate model size
    model.save(f"models/h5/temp_{model_name}.h5")
    h5_size = os.path.getsize(f"models/h5/temp_{model_name}.h5") / (1024 * 1024)
    print(f"Model size: {h5_size:.2f} MB")
    
    # Delete temp file
    if os.path.exists(f"models/h5/temp_{model_name}.h5"):
        os.remove(f"models/h5/temp_{model_name}.h5")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'size_mb': h5_size
    }

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                print("Could not set memory growth")
    
    # Load and preprocess data
    scaled_data = load_and_preprocess_data()
    
    # Models to train
    models = {
        'QCNN': create_qcnn,
        'TinyLSTM': create_lstm,
        'CarbonMLP': create_mlp,
        'FBNet': create_fbnet
    }
    
    # Store results
    results = {}
    efficiency_metrics = {}
    
    # Train and evaluate each model
    for model_name, create_fn in models.items():
        results[model_name] = {}
        print(f"\n=== Training {model_name} ===")
        
        # Train model
        model, history, train_time = train_model(model_name, create_fn, scaled_data)
        
        # Analyze model efficiency
        efficiency_metrics[model_name] = analyze_model_efficiency(model, model_name)
        efficiency_metrics[model_name]['training_time'] = train_time
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Save model
        model.save(f"models/h5/union_{model_name}.h5")
        print(f"Model saved: models/h5/union_{model_name}.h5")
        
        # Convert to TFLite
        tflite_size = convert_to_tflite(model, model_name, scaled_data)
        efficiency_metrics[model_name]['tflite_size_kb'] = tflite_size
        
        # Evaluate model with all requested metrics
        metrics, y_pred = evaluate_model(model, scaled_data['X_test'], scaled_data['y_test'])
        
        # Save results
        results[model_name] = {
            'metrics': metrics,
            'train_time': train_time
        }
        
        # Print metrics
        print(f"\nResults for {model_name}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(scaled_data['y_test'], y_pred)
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{model_name} Confusion Matrix\nAccuracy: {metrics["accuracy"]:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'plots/union_{model_name}_cm.png')
        plt.close()
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Save results
    import pickle
    with open('results/union_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save efficiency metrics
    with open('results/efficiency_metrics.pkl', 'wb') as f:
        pickle.dump(efficiency_metrics, f)
    
    # Print summary
    print("\n=== Model Performance and Efficiency Summary ===")
    print(f"{'Model':<10} {'Accuracy':<10} {'F1':<10} {'Params':<12} {'Size (MB)':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for model_name in models.keys():
        metrics = results[model_name]['metrics']
        efficiency = efficiency_metrics[model_name]
        print(f"{model_name:<10} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
              f"{efficiency['total_params']:<12,} {efficiency['size_mb']:<10.2f} "
              f"{efficiency['training_time']:<10.2f}")
    
    print("\nTraining complete! All models and results saved.")

if __name__ == "__main__":
    main()
