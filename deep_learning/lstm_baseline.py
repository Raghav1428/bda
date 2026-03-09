"""
lstm_baseline.py
================
Non-distributed LSTM baseline for AQI forecasting.

Converts Spark DataFrame to Pandas, creates time-series sequences,
trains a 2-layer LSTM model using TensorFlow/Keras, and evaluates RMSE.

This serves as a centralized baseline for comparison with distributed
PySpark MLlib models.

NOTE: The label column is 'AQI_target' (next-day AQI), matching the
distributed models' forecasting formulation.
"""

import os
import time
import warnings
import numpy as np

# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Reproducibility seed
SEED = 42


def spark_to_pandas(df, feature_columns, label_col="AQI_target"):
    """
    Convert a Spark DataFrame to a Pandas DataFrame with selected columns.

    Args:
        df (DataFrame): Spark DataFrame with features.
        feature_columns (list): List of feature column names.
        label_col (str): Label column name.

    Returns:
        tuple: (X numpy array, y numpy array)
    """
    cols_to_select = feature_columns + [label_col]
    # Only select columns that exist in the dataframe
    existing_cols = [c for c in cols_to_select if c in df.columns]
    pdf = df.select(existing_cols).toPandas()

    X = pdf[feature_columns].values.astype(np.float32)
    y = pdf[label_col].values.astype(np.float32)

    print(f"[LSTM] Converted to Pandas: X shape={X.shape}, y shape={y.shape}")
    return X, y


def create_sequences(X, y, window_size=24):
    """
    Create sliding-window sequences for LSTM input.

    Args:
        X (np.ndarray): Feature array of shape (n_samples, n_features).
        y (np.ndarray): Label array of shape (n_samples,).
        window_size (int): Number of time steps per sequence.

    Returns:
        tuple: (X_seq of shape (n_seq, window_size, n_features),
                y_seq of shape (n_seq,))
    """
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    print(f"[LSTM] Created sequences: X_seq shape={X_seq.shape}, y_seq shape={y_seq.shape}")
    return X_seq, y_seq


def build_lstm_model(input_shape):
    """
    Build a 2-layer LSTM model for AQI regression.

    Architecture:
      - LSTM(64, return_sequences=True)
      - LSTM(32)
      - Dense(1)

    Args:
        input_shape (tuple): Shape of input (window_size, n_features).

    Returns:
        keras.Model: Compiled LSTM model.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    # Set seeds for reproducibility
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    model.summary()
    return model


def train_lstm(df, feature_columns, label_col="AQI_target", window_size=24, epochs=20, batch_size=32):
    """
    Full LSTM training pipeline: convert data → create sequences → train → evaluate.

    Args:
        df (DataFrame): Spark DataFrame with features.
        feature_columns (list): Feature column names (raw, not assembled).
        label_col (str): Label column name (default: AQI_target for next-day).
        window_size (int): Sequence length.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        dict: {"model": "LSTM", "RMSE": val, "MAE": val, "R2": val, "training_time_sec": val}
    """
    print("\n" + "=" * 60)
    print(" LSTM BASELINE TRAINING (Non-Distributed)")
    print("=" * 60)

    try:
        import tensorflow as tf
    except ImportError:
        print("[ERROR] TensorFlow not installed. Skipping LSTM baseline.")
        print("[ERROR] Install with: pip install tensorflow")
        return {
            "model": "LSTM",
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "R2": float("nan"),
            "training_time_sec": 0.0
        }

    # Step 1: Convert to Pandas
    X, y = spark_to_pandas(df, feature_columns, label_col)

    # Step 2: Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Step 3: Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

    # Step 4: Train/test split (80/20)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"[LSTM] Train: {X_train.shape[0]} sequences | Test: {X_test.shape[0]} sequences")

    # Step 5: Build model
    model = build_lstm_model(input_shape=(window_size, X.shape[1]))

    # Step 6: Train
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    elapsed = time.time() - start
    print(f"[LSTM] Training completed in {elapsed:.2f}s")

    # Step 7: Evaluate
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform predictions and actuals
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Compute metrics
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    mae = np.mean(np.abs(y_actual - y_pred))

    # MAPE — exclude zeros to avoid division by zero
    nonzero_mask = y_actual != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_actual[nonzero_mask] - y_pred[nonzero_mask])
                              / y_actual[nonzero_mask])) * 100
    else:
        mape = float("nan")

    # R² score
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print(f"\n[LSTM] Evaluation Results:")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    R²   : {r2:.4f}")
    print(f"    Time : {elapsed:.2f}s")
    print("=" * 60 + "\n")

    return {
        "model": "LSTM",
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
        "training_time_sec": round(elapsed, 2)
    }


if __name__ == "__main__":
    print("lstm_baseline.py — Run via main.py for full pipeline execution.")
