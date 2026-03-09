"""
feature_engineering.py
======================
Distributed feature engineering for AQI forecasting.

NEXT-DAY FORECASTING DESIGN (Scientific Rationale)
----------------------------------------------------
Why predicting same-day AQI from same-day pollutants is deterministic:
  AQI is COMPUTED from pollutant concentrations using the CPCB breakpoint
  formula: AQI = max(I_PM2.5, I_PM10, I_NO2, I_SO2).  If the model
  receives today's PM2.5, PM10, SO2, NO2 as features and today's AQI as
  the label, it is essentially learning a known mathematical formula.
  Tree-based models can approximate piecewise-linear functions almost
  perfectly, leading to artificially inflated R² (>0.99).  This does NOT
  constitute genuine forecasting.

Why next-day AQI prediction is scientifically valid:
  By using lead(AQI, 1) as the target, we predict TOMORROW's AQI using
  TODAY's observations.  The model must learn temporal dynamics —
  persistence, trend evolution, and seasonal patterns — rather than a
  deterministic formula.  This is the standard formulation for short-term
  air quality forecasting in the environmental science literature.

Expected impact on R²:
  R² will decrease substantially (expected range: 0.3–0.7 depending on
  the city and season).  This is MORE HONEST because it reflects genuine
  forecasting skill rather than formula reconstruction.  A lower R² from
  a correct formulation is scientifically superior to a higher R² from a
  leaked formulation.

Features created:
  - Target: AQI_target (next-day AQI via lead)
  - Lag features: AQI_lag1, AQI_lag24 (backward-looking only)
  - Rolling mean: AQI_rolling_24 (24-row backward window)
    NOTE: Uses 24-row window (not 24-hour) because the dataset does not
          guarantee hourly frequency.
  - Cyclic encoding: sin(month), cos(month)
  - Interaction features: PM2.5 × SO2, NO2 × PM10
  - Feature vector assembled via VectorAssembler
"""

import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler


def create_forecast_target(df, partition_col="city", order_col="date_parsed"):
    """
    Create the next-day AQI prediction target using lead().

    AQI_target(t) = AQI(t+1)

    This converts the problem from "reconstruct today's AQI from today's
    pollutants" to "predict tomorrow's AQI from today's data".

    Rows where AQI_target is null (last row per city) are dropped because
    there is no future value to predict.

    IMPORTANT: Only lead() is used for the TARGET.  All FEATURES use
    lag() or backward-looking windows — no future information leaks into
    the feature set.

    Args:
        df (DataFrame): Preprocessed dataset with AQI column.
        partition_col (str): Column to partition by (city).
        order_col (str): Column to order by (date).

    Returns:
        DataFrame: Dataset with AQI_target column, null targets dropped.
    """
    window_spec = Window.partitionBy(partition_col).orderBy(order_col)

    before = df.count()
    df = df.withColumn("AQI_target", F.lead("AQI", 1).over(window_spec))

    # Drop rows where we have no future AQI to predict
    df = df.filter(F.col("AQI_target").isNotNull())
    after = df.count()

    print(f"[INFO] Created AQI_target = lead(AQI, 1) — next-day forecasting target")
    print(f"[INFO] Dropped {before - after} rows with no future AQI ({after} remaining)")
    return df


def add_lag_features(df, partition_col="city", order_col="date_parsed"):
    """
    Add lag features for AQI using Spark Window functions.

    Creates:
      - AQI_lag1:  AQI value 1 row prior (within same city)
      - AQI_lag24: AQI value 24 rows prior (within same city)

    NOTE: These features use lag() (backward-looking) — they NEVER
    access future data.  This is critical for preventing temporal leakage.

    Args:
        df (DataFrame): Preprocessed dataset.
        partition_col (str): Column to partition by (city).
        order_col (str): Column to order by (date).

    Returns:
        DataFrame: Dataset with lag features.
    """
    window_spec = Window.partitionBy(partition_col).orderBy(order_col)

    df = df.withColumn("AQI_lag1", F.lag("AQI", 1).over(window_spec))
    df = df.withColumn("AQI_lag24", F.lag("AQI", 24).over(window_spec))

    print("[INFO] Added lag features: AQI_lag1, AQI_lag24")
    return df


def add_rolling_mean(df, partition_col="city", order_col="date_parsed", window_size=24):
    """
    Add a rolling mean of AQI over the last `window_size` rows.

    NOTE: This is a 24-ROW BACKWARD-LOOKING rolling window.
    rowsBetween(-(window_size - 1), 0) ensures the window includes only
    past and current rows — NO future rows are included.

    Args:
        df (DataFrame): Dataset with AQI column.
        partition_col (str): Column to partition by.
        order_col (str): Column to order by.
        window_size (int): Number of rows for rolling window.

    Returns:
        DataFrame: Dataset with AQI_rolling_24 column.
    """
    window_spec = (
        Window.partitionBy(partition_col)
        .orderBy(order_col)
        .rowsBetween(-(window_size - 1), 0)
    )

    df = df.withColumn("AQI_rolling_24", F.avg("AQI").over(window_spec))

    print(f"[INFO] Added rolling mean: AQI_rolling_24 ({window_size}-row backward window)")
    return df


def add_cyclic_encoding(df):
    """
    Add cyclic encoding for the month feature to capture seasonal periodicity.

    Creates:
      - month_sin: sin(2π × month / 12)
      - month_cos: cos(2π × month / 12)

    Args:
        df (DataFrame): Dataset with 'month' column.

    Returns:
        DataFrame: Dataset with cyclic month features.
    """
    df = df.withColumn(
        "month_sin",
        F.sin(2 * math.pi * F.col("month") / 12)
    )
    df = df.withColumn(
        "month_cos",
        F.cos(2 * math.pi * F.col("month") / 12)
    )

    print("[INFO] Added cyclic encoding: month_sin, month_cos")
    return df


def add_interaction_features(df):
    """
    Add interaction features between pollutants.

    Creates:
      - pm25_x_so2:  PM2.5 × SO2
      - no2_x_pm10:  NO2 × PM10

    Note: The original dataset lacks Humidity and Temperature columns.
    These interaction features substitute PM2.5×Humidity and NO2×Temperature.

    Args:
        df (DataFrame): Dataset with pollutant columns.

    Returns:
        DataFrame: Dataset with interaction features.
    """
    df = df.withColumn("pm25_x_so2", F.col("pm2_5") * F.col("so2"))
    df = df.withColumn("no2_x_pm10", F.col("no2") * F.col("pm10"))

    print("[INFO] Added interaction features: pm25_x_so2, no2_x_pm10")
    return df


def assemble_features(df, scale=True):
    """
    Assemble all engineered features into a single 'features' vector column
    using VectorAssembler. Optionally apply StandardScaler.

    Feature columns assembled:
      so2, no2, pm10, pm2_5, month, day,
      AQI_lag1, AQI_lag24, AQI_rolling_24,
      month_sin, month_cos, pm25_x_so2, no2_x_pm10

    NOTE: same-day pollutants (so2, no2, pm10, pm2_5) are RETAINED as
    features because they are observations at time t, and the target is
    AQI at time t+1.  This models the persistence-based evolution of
    pollutant concentrations into next-day AQI.

    Args:
        df (DataFrame): Dataset with all feature columns.
        scale (bool): If True, apply StandardScaler after assembly.

    Returns:
        tuple: (DataFrame with 'features' column, list of feature names)
    """
    feature_columns = [
        "so2", "no2", "pm10", "pm2_5",
        "month", "day",
        "AQI_lag1", "AQI_lag24", "AQI_rolling_24",
        "month_sin", "month_cos",
        "pm25_x_so2", "no2_x_pm10"
    ]

    # Drop rows with nulls in feature columns (from lag/rolling operations)
    before = df.count()
    df = df.dropna(subset=feature_columns + ["AQI_target"])
    after = df.count()
    print(f"[INFO] Dropped {before - after} rows with null features ({after} remaining)")

    # Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    df = assembler.transform(df)

    if scale:
        # Apply StandardScaler (important for Linear Regression fairness)
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        print("[INFO] Applied StandardScaler to feature vector")
    else:
        df = df.withColumnRenamed("features_raw", "features")

    print(f"[INFO] Assembled {len(feature_columns)} features into 'features' vector")
    return df, feature_columns


def run_feature_engineering(df, scale=True):
    """
    Execute the full feature engineering pipeline.

    Pipeline:
      1. Create next-day forecast target (AQI_target)
      2. Add lag features (AQI_lag1, AQI_lag24)
      3. Add rolling mean (AQI_rolling_24)
      4. Add cyclic encoding (month_sin, month_cos)
      5. Add interaction features (pm25_x_so2, no2_x_pm10)
      6. Assemble and optionally scale features

    Args:
        df (DataFrame): Preprocessed dataset.
        scale (bool): Whether to apply StandardScaler.

    Returns:
        tuple: (DataFrame with features, list of feature names)
    """
    print("\n" + "=" * 60)
    print(" FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # --- NEXT-DAY FORECASTING TARGET ---
    df = create_forecast_target(df)

    df = add_lag_features(df)
    df = add_rolling_mean(df)
    df = add_cyclic_encoding(df)
    df = add_interaction_features(df)
    df, feature_names = assemble_features(df, scale=scale)

    print("=" * 60 + "\n")
    return df, feature_names


if __name__ == "__main__":
    from spark_pipeline.data_ingestion import create_spark_session, load_data
    from spark_pipeline.preprocessing import run_preprocessing

    spark = create_spark_session()
    raw_df = load_data(spark, local_path="data/india_air_quality.csv")
    clean_df = run_preprocessing(raw_df)
    feat_df, feat_names = run_feature_engineering(clean_df)
    print("Feature columns:", feat_names)
    feat_df.select("city", "AQI", "AQI_target", "features").show(5, truncate=False)
    spark.stop()
