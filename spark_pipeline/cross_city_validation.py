"""
cross_city_validation.py
========================
Leave-One-City-Out cross-city generalization validation.

Strategy:
  - For each target city:
      1. Train on ALL data except that city
      2. Test on the held-out city
      3. Evaluate RMSE, MAE, R²
  - Loop over the top 3 cities by data volume

This tests the model's ability to generalize to unseen cities
(spatial generalization), a key research contribution.

NOTE: The label column is 'AQI_target' (next-day AQI), reflecting
the scientifically valid forecasting formulation.
"""

import time
from pyspark.sql import DataFrame
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# Reproducibility seed
SEED = 42

# Label column — next-day AQI
LABEL_COL = "AQI_target"


def get_top_cities(df, city_col="city", n=3):
    """
    Get the top N cities by data volume (row count).

    Args:
        df (DataFrame): Dataset with city column.
        city_col (str): Name of the city column.
        n (int): Number of cities to select.

    Returns:
        list[str]: List of top N city names.
    """
    city_counts = (
        df.groupBy(city_col)
        .count()
        .orderBy("count", ascending=False)
        .limit(n)
        .collect()
    )
    cities = [row[city_col] for row in city_counts]
    print(f"[INFO] Top {n} cities by data volume: {cities}")
    return cities


def leave_one_city_out(df, target_city, city_col="city"):
    """
    Split data into train (all cities except target) and test (target city only).

    Args:
        df (DataFrame): Full feature-engineered dataset.
        target_city (str): City to hold out for testing.
        city_col (str): Name of the city column.

    Returns:
        tuple: (train_df, test_df)
    """
    train_df = df.filter(df[city_col] != target_city)
    test_df = df.filter(df[city_col] == target_city)
    return train_df, test_df


def evaluate_predictions(predictions, model_name="Model"):
    """
    Evaluate predictions using RMSE, MAE, R².

    Args:
        predictions (DataFrame): Predictions with LABEL_COL and 'prediction' columns.
        model_name (str): Display name.

    Returns:
        dict: Metric dictionary.
    """
    rmse = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse"
    ).evaluate(predictions)

    mae = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="mae"
    ).evaluate(predictions)

    r2 = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="r2"
    ).evaluate(predictions)

    return {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}


def run_cross_city_validation(df, n_cities=3):
    """
    Execute leave-one-city-out validation using RandomForest.

    For each of the top N cities:
      - Train on all other cities
      - Test on the held-out city
      - Record RMSE, MAE, R²

    Args:
        df (DataFrame): Full feature-engineered dataset.
        n_cities (int): Number of cities to evaluate.

    Returns:
        list[dict]: Per-city evaluation results.
    """
    print("\n" + "=" * 60)
    print(" CROSS-CITY VALIDATION (Leave-One-City-Out)")
    print("=" * 60)

    cities = get_top_cities(df, n=n_cities)
    results = []

    for city in cities:
        print(f"\n--- Held-out city: {city} ---")

        train_df, test_df = leave_one_city_out(df, city)
        train_count = train_df.count()
        test_count = test_df.count()
        print(f"    Train: {train_count} rows | Test: {test_count} rows")

        if test_count == 0:
            print(f"    [WARN] No test data for city '{city}'. Skipping.")
            continue

        # Train Random Forest on remaining cities
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol=LABEL_COL,
            numTrees=100,
            maxDepth=10,
            seed=SEED
        )

        start = time.time()
        model = rf.fit(train_df)
        elapsed = time.time() - start

        # Evaluate on held-out city
        predictions = model.transform(test_df)
        metrics = evaluate_predictions(predictions, model_name=f"RF_cross_{city}")
        metrics["city"] = city
        metrics["training_time_sec"] = round(elapsed, 2)
        results.append(metrics)

        print(f"    RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | "
              f"R²: {metrics['R2']:.4f} | Time: {elapsed:.2f}s")

    # Summary table
    print("\n" + "-" * 70)
    print(" Cross-City Validation Summary")
    print("-" * 70)
    header = f"{'City':20s} | {'RMSE':>10s} | {'MAE':>10s} | {'R²':>10s}"
    print(header)
    print("-" * 70)
    for r in results:
        print(f"{r['city']:20s} | {r['RMSE']:10.4f} | {r['MAE']:10.4f} | {r['R2']:10.4f}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    print("cross_city_validation.py — Run via main.py for full pipeline execution.")
