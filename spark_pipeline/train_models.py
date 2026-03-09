"""
train_models.py
===============
Distributed model training using PySpark MLlib.

Trains and compares:
  - LinearRegression
  - RandomForestRegressor
  - GBTRegressor

TEMPORAL SPLITTING (Scientific Rationale)
-----------------------------------------
Why random split causes temporal leakage:
  Random split allows future observations to leak into the training set.
  Because AQI exhibits strong temporal autocorrelation (today's AQI is
  correlated with yesterday's), a model trained on random samples can
  "cheat" by memorising temporal neighbors of test points.  This inflates
  R² and produces overly optimistic results that do NOT reflect real-world
  forecasting skill where the model has never seen the future.

This module implements STRICT CHRONOLOGICAL splitting:
  • Compute the full range of years present in the data.
  • Allocate the earliest ~80% of years to training, the latest ~20% to
    testing.
  • If the data spans only a single year, sort by date within each city
    and split at the 80th-percentile date.
  • No shuffling is ever performed.

All models use seed=42 for reproducibility.
Training time is logged for each model (scalability analysis).
"""

import time
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)


# Reproducibility seed
SEED = 42

# Label column — next-day AQI (see feature_engineering.py)
LABEL_COL = "AQI_target"


def chronological_split(df, train_ratio=0.8, min_test_years=2):
    """
    Split the dataset chronologically — NO random splitting, NO shuffling.

    Strategy:
      1. Collect all unique years and their row counts, sorted ascending.
      2. If multiple years exist:
           - Walk through years from earliest to latest, accumulating rows.
           - Once the cumulative row count reaches ~train_ratio of total
             rows, draw the boundary there.
           - Enforce min_test_years: even if the row ratio is satisfied
             early, always keep at least min_test_years years in the test
             set.  This ensures adequate temporal coverage for evaluation.
      3. If only one year exists:
           - Compute the date at the train_ratio percentile.
           - Train on rows before that date, test on rows after.
      4. Print comprehensive split diagnostics.

    Data limitation note:
      The India Air Quality dataset spans 1987–2015.  With min_test_years=2,
      the test set covers 2014–2015.  Post-2015 data is not available in
      this dataset; future work should evaluate on more recent observations.

    Args:
        df (DataFrame): Feature-engineered dataset with 'year' and
                        'date_parsed' columns.
        train_ratio (float): Target fraction of ROWS for training.
        min_test_years (int): Minimum number of years in the test set.

    Returns:
        tuple: (train_df, test_df)
    """
    # --- Collect unique years with row counts ---
    year_counts = (
        df.groupBy("year")
        .count()
        .orderBy("year")
        .collect()
    )
    years = [row["year"] for row in year_counts]
    counts = [row["count"] for row in year_counts]
    n_years = len(years)
    total_rows = sum(counts)

    print(f"[INFO] Unique years in dataset: {years[0]}–{years[-1]} ({n_years} years)")
    print(f"[INFO] Total rows: {total_rows}")

    if n_years > 1:
        # --- Multi-year split: find year boundary by ROW COUNT ---
        # Enforce: at least min_test_years years must be in test set
        max_split_idx = n_years - min_test_years - 1

        cumulative = 0
        split_idx = 0
        for i, cnt in enumerate(counts):
            cumulative += cnt
            if cumulative / total_rows >= train_ratio:
                split_idx = i
                break
        else:
            split_idx = n_years - 2

        # Clamp so that at least min_test_years remain for testing
        split_idx = min(split_idx, max_split_idx)

        train_years = years[:split_idx + 1]
        test_years = years[split_idx + 1:]

        train_df = df.filter(F.col("year").isin(train_years))
        test_df = df.filter(F.col("year").isin(test_years))

        print(f"[INFO] Chronological split (target row ratio={train_ratio}, "
              f"min_test_years={min_test_years})")
        print(f"[INFO]   Train years: {train_years[0]}–{train_years[-1]} "
              f"({len(train_years)} years)")
        print(f"[INFO]   Test  years: {test_years[0]}–{test_years[-1]} "
              f"({len(test_years)} years)")

    else:
        # --- Single-year split: split by date ---
        date_quantile = df.select(
            F.percentile_approx("date_parsed", train_ratio).alias("split_date")
        ).first()["split_date"]

        train_df = df.filter(F.col("date_parsed") <= date_quantile)
        test_df = df.filter(F.col("date_parsed") > date_quantile)

        print(f"[INFO] Chronological split by DATE (single year: {years[0]})")
        print(f"[INFO]   Split date: {date_quantile}")

    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count
    actual_ratio = train_count / total if total > 0 else 0

    print(f"[INFO]   Train: {train_count} rows ({actual_ratio:.1%})")
    print(f"[INFO]   Test:  {test_count} rows ({1 - actual_ratio:.1%})")

    return train_df, test_df


def train_linear_regression(train_df):
    """
    Train a LinearRegression model.

    Args:
        train_df (DataFrame): Training data with 'features' and LABEL_COL.

    Returns:
        tuple: (trained model, training time in seconds)
    """
    print("\n[MODEL] Training LinearRegression...")
    lr = LinearRegression(
        featuresCol="features",
        labelCol=LABEL_COL,
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.8
    )

    start = time.time()
    model = lr.fit(train_df)
    elapsed = time.time() - start

    print(f"[MODEL] LinearRegression trained in {elapsed:.2f}s")
    print(f"[MODEL]   Coefficients: {model.coefficients[:5]}... (showing first 5)")
    print(f"[MODEL]   Intercept: {model.intercept:.4f}")
    return model, elapsed


def train_random_forest(train_df):
    """
    Train a RandomForestRegressor model.

    Args:
        train_df (DataFrame): Training data with 'features' and LABEL_COL.

    Returns:
        tuple: (trained model, training time in seconds)
    """
    print("\n[MODEL] Training RandomForestRegressor...")
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

    print(f"[MODEL] RandomForestRegressor trained in {elapsed:.2f}s")
    return model, elapsed


def train_gbt(train_df):
    """
    Train a Gradient-Boosted Tree Regressor model.

    Args:
        train_df (DataFrame): Training data with 'features' and LABEL_COL.

    Returns:
        tuple: (trained model, training time in seconds)
    """
    print("\n[MODEL] Training GBTRegressor...")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=LABEL_COL,
        maxIter=100,
        maxDepth=5,
        seed=SEED
    )

    start = time.time()
    model = gbt.fit(train_df)
    elapsed = time.time() - start

    print(f"[MODEL] GBTRegressor trained in {elapsed:.2f}s")
    return model, elapsed


def train_all_models(train_df):
    """
    Train all three models and collect results.

    Args:
        train_df (DataFrame): Training data.

    Returns:
        dict: {model_name: (model, training_time_seconds)}
    """
    print("\n" + "=" * 60)
    print(" MODEL TRAINING")
    print("=" * 60)

    results = {}

    # Train each model
    lr_model, lr_time = train_linear_regression(train_df)
    results["LinearRegression"] = (lr_model, lr_time)

    rf_model, rf_time = train_random_forest(train_df)
    results["RandomForest"] = (rf_model, rf_time)

    gbt_model, gbt_time = train_gbt(train_df)
    results["GBTRegressor"] = (gbt_model, gbt_time)

    print("\n" + "-" * 40)
    print(" Training Time Summary")
    print("-" * 40)
    for name, (_, t) in results.items():
        print(f"  {name:25s} : {t:.2f}s")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    from spark_pipeline.data_ingestion import create_spark_session, load_data
    from spark_pipeline.preprocessing import run_preprocessing
    from spark_pipeline.feature_engineering import run_feature_engineering

    spark = create_spark_session()
    raw_df = load_data(spark, local_path="data/india_air_quality.csv")
    clean_df = run_preprocessing(raw_df)
    feat_df, feat_names = run_feature_engineering(clean_df)
    train_df, test_df = chronological_split(feat_df)
    models = train_all_models(train_df)
    spark.stop()
