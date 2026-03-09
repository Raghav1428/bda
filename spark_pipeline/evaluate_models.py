"""
evaluate_models.py
==================
Model evaluation module for AQI forecasting.

Computes:
  - RMSE (Root Mean Squared Error)
  - MAE  (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - R²   (Coefficient of Determination)

Also extracts and saves feature importance from Random Forest.
All results are saved to the results/ directory.

NOTE: The label column is 'AQI_target' (next-day AQI), not 'AQI'.
This reflects the next-day forecasting formulation where the model
predicts tomorrow's AQI from today's observations.

Why lower R² after this correction is more honest:
  In the previous formulation, R² > 0.99 because the model was
  reconstructing a known deterministic formula (CPCB breakpoints).
  With next-day prediction, R² will decrease because the model must
  now learn genuine temporal dynamics.  A lower R² (e.g., 0.3–0.7)
  from an honest formulation is scientifically MORE VALUABLE than a
  near-perfect R² from formula reconstruction.
"""

import os
import csv
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, functions as F


# Label column — next-day AQI
LABEL_COL = "AQI_target"


def evaluate_model(model, test_df, model_name="Model"):
    """
    Evaluate a trained model on the test set using RMSE, MAE, MAPE, and R².

    Args:
        model: Trained PySpark ML model.
        test_df (DataFrame): Test data with 'features' and LABEL_COL columns.
        model_name (str): Display name for the model.

    Returns:
        dict: {"model": name, "RMSE": val, "MAE": val, "MAPE": val, "R2": val}
    """
    predictions = model.transform(test_df)

    # RMSE
    rmse_eval = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse"
    )
    rmse = rmse_eval.evaluate(predictions)

    # MAE
    mae_eval = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="mae"
    )
    mae = mae_eval.evaluate(predictions)

    # R²
    r2_eval = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="r2"
    )
    r2 = r2_eval.evaluate(predictions)

    # MAPE — computed manually (not in PySpark's RegressionEvaluator)
    # MAPE = (1/N) * Σ |actual - predicted| / |actual| * 100
    # Exclude rows where actual == 0 to avoid division by zero.
    mape = predictions.filter(F.col(LABEL_COL) != 0.0).select(
        F.mean(F.abs(F.col(LABEL_COL) - F.col("prediction")) / F.abs(F.col(LABEL_COL))) * 100
    ).first()[0]
    mape = mape if mape is not None else float("nan")

    print(f"\n  [{model_name}] Evaluation Results:")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    R²   : {r2:.4f}")

    return {"model": model_name, "RMSE": rmse, "MAE": mae, "MAPE": round(mape, 2), "R2": r2}


def evaluate_all_models(trained_models, test_df):
    """
    Evaluate all trained models and collect metrics.

    Also prints test-set AQI_target statistics (mean, std) so readers
    can interpret error magnitudes relative to the target scale.

    Args:
        trained_models (dict): {name: (model, train_time)} from train_all_models.
        test_df (DataFrame): Test data.

    Returns:
        list[dict]: List of metric dictionaries for each model.
    """
    print("\n" + "=" * 60)
    print(" MODEL EVALUATION")
    print("=" * 60)

    # --- Test set target statistics (for error scale context) ---
    stats = test_df.select(
        F.mean(LABEL_COL).alias("mean"),
        F.stddev(LABEL_COL).alias("std"),
        F.min(LABEL_COL).alias("min"),
        F.max(LABEL_COL).alias("max"),
    ).first()

    print(f"\n  Test set AQI_target statistics:")
    print(f"    Mean : {stats['mean']:.2f}")
    print(f"    Std  : {stats['std']:.2f}")
    print(f"    Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    print(f"  (Use Std to gauge error scale: RMSE/Std ≈ normalised error)")

    all_metrics = []
    for name, (model, train_time) in trained_models.items():
        metrics = evaluate_model(model, test_df, model_name=name)
        metrics["training_time_sec"] = round(train_time, 2)
        all_metrics.append(metrics)

    print("\n" + "=" * 60)
    return all_metrics


def extract_feature_importance(rf_model, feature_names, output_path="results/feature_importance.csv"):
    """
    Extract and save ranked feature importance from a Random Forest model.

    Args:
        rf_model: Trained RandomForestRegressionModel.
        feature_names (list): List of feature column names.
        output_path (str): Path to save the CSV file.

    Returns:
        list[tuple]: Sorted list of (feature_name, importance) tuples.
    """
    importances = rf_model.featureImportances.toArray()

    # Pair feature names with their importance values
    feature_imp = list(zip(feature_names, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for feat, imp in feature_imp:
            writer.writerow([feat, round(imp, 6)])

    print(f"\n[INFO] Feature importance saved to: {output_path}")
    print("[INFO] Top 5 features:")
    for feat, imp in feature_imp[:5]:
        print(f"    {feat:20s} : {imp:.6f}")

    return feature_imp


def save_metrics_to_csv(all_metrics, output_path="results/metrics_output.csv"):
    """
    Save all model metrics (including LSTM) to a CSV file.

    Args:
        all_metrics (list[dict]): List of metric dictionaries.
        output_path (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["model", "RMSE", "MAE", "MAPE", "R2", "training_time_sec"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in all_metrics:
            row = {k: metrics.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\n[INFO] All metrics saved to: {output_path}")


def print_comparison_table(all_metrics):
    """
    Print a formatted comparison table of all models.

    Args:
        all_metrics (list[dict]): List of metric dictionaries.
    """
    print("\n" + "=" * 90)
    print(" MODEL COMPARISON TABLE")
    print("=" * 90)
    header = (
        f"{'Model':25s} | {'RMSE':>10s} | {'MAE':>10s} | {'MAPE (%)':>10s} | "
        f"{'R²':>10s} | {'Time (s)':>10s}"
    )
    print(header)
    print("-" * 90)
    for m in all_metrics:
        row = (
            f"{m['model']:25s} | "
            f"{m['RMSE']:10.4f} | "
            f"{m['MAE']:10.4f} | "
            f"{m.get('MAPE', 0):10.2f} | "
            f"{m['R2']:10.4f} | "
            f"{m.get('training_time_sec', 0):10.2f}"
        )
        print(row)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    print("evaluate_models.py — Run via main.py for full pipeline execution.")
