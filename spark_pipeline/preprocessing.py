"""
preprocessing.py
=================
Distributed preprocessing for the India Air Quality dataset.

Steps:
  1. Rename rspm → pm10 (RSPM = PM10 in older Indian datasets)
  2. Cast string columns to numeric (handles literal 'NA' values)
  3. Impute missing pollutant values with column-wise median
  4. Compute AQI using official CPCB sub-index breakpoints (max sub-index rule)
     — uses pure Spark SQL expressions (no Python UDF) for reliability
  5. Drop rows where AQI cannot be computed (all pollutants missing)
  6. Remove outliers using IQR method on AQI
  7. Parse date column → timestamp; extract year, month, day
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


# ============================================================================
# CPCB AQI Breakpoints — Indian National Air Quality Index
# Each entry: (BPlo, BPhi, Ilo, Ihi)
# ============================================================================
PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (250, 500, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 250, 101, 200),
    (251, 350, 201, 300),
    (351, 430, 301, 400),
    (430, 600, 401, 500),
]

SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 380, 101, 200),
    (381, 800, 201, 300),
    (801, 1600, 301, 400),
    (1600, 2400, 401, 500),
]

NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 180, 101, 200),
    (181, 280, 201, 300),
    (281, 400, 301, 400),
    (400, 600, 401, 500),
]


def _build_sub_index_expr(col_name, breakpoints):
    """
    Build a pure Spark SQL expression for computing a sub-index.

    Uses F.when/F.otherwise chains — runs natively in JVM, no Python UDF needed.
    Formula:  I = ((Ihi - Ilo) / (BPhi - BPlo)) * (C - BPlo) + Ilo

    Args:
        col_name (str): Column name for the pollutant.
        breakpoints (list): List of (BPlo, BPhi, Ilo, Ihi) tuples.

    Returns:
        Column: Spark SQL expression that evaluates to the sub-index.
    """
    col = F.col(col_name)
    expr = F.lit(None).cast("double")

    # Build chain in reverse so first matching breakpoint wins
    for bp_lo, bp_hi, i_lo, i_hi in reversed(breakpoints):
        slope = (i_hi - i_lo) / (bp_hi - bp_lo) if bp_hi != bp_lo else 0.0
        expr = F.when(
            (col >= bp_lo) & (col <= bp_hi),
            F.lit(slope) * (col - F.lit(bp_lo)) + F.lit(float(i_lo))
        ).otherwise(expr)

    # Cap at 500 if concentration exceeds all breakpoints
    max_bp = breakpoints[-1][1]
    expr = F.when(col > max_bp, F.lit(500.0)).otherwise(expr)

    return expr


def rename_columns(df):
    """
    Rename rspm → pm10 (RSPM is PM10 in older Indian monitoring datasets).
    Also standardize column names to lowercase with underscores.

    Args:
        df (DataFrame): Raw dataset.

    Returns:
        DataFrame: Dataset with renamed columns.
    """
    df = df.withColumnRenamed("rspm", "pm10")
    print("[INFO] Renamed 'rspm' → 'pm10' (RSPM = PM10 in Indian datasets)")
    return df


def cast_numeric_columns(df, columns=None):
    """
    Cast pollutant columns from StringType to DoubleType.
    The raw CSV contains literal 'NA' strings which cause Spark to infer
    these columns as StringType. We first replace 'NA' and empty strings
    with null, then safely cast to double.

    Args:
        df (DataFrame): Dataset with string-typed pollutant columns.
        columns (list): Columns to cast. Defaults to pollutant + spm columns.

    Returns:
        DataFrame: Dataset with numeric pollutant columns.
    """
    if columns is None:
        columns = ["so2", "no2", "pm10", "pm2_5", "spm"]

    for col_name in columns:
        if col_name in df.columns:
            col_type = dict(df.dtypes).get(col_name, "")
            if col_type == "string":
                # Step 1: Replace 'NA', 'None', empty, whitespace-only → null
                df = df.withColumn(
                    col_name,
                    F.when(
                        (F.trim(F.col(col_name)) == "") |
                        (F.upper(F.trim(F.col(col_name))) == "NA") |
                        (F.upper(F.trim(F.col(col_name))) == "NONE"),
                        F.lit(None)
                    ).otherwise(F.col(col_name))
                )
                # Step 2: Now cast clean values to double
                df = df.withColumn(col_name, F.col(col_name).cast("double"))
                print(f"[INFO] Cast '{col_name}' from StringType → DoubleType (NA → null)")

    return df


def compute_aqi_column(df):
    """
    Compute AQI from pollutant sub-indices using CPCB breakpoints.
    AQI = max(I_PM2.5, I_PM10, I_NO2, I_SO2)

    Uses pure Spark SQL expressions — no Python UDF needed.

    Args:
        df (DataFrame): Dataset with pm2_5, pm10, so2, no2 columns.

    Returns:
        DataFrame: Dataset with new 'AQI' column.
    """
    # Build sub-index expressions for each pollutant
    pm25_si = _build_sub_index_expr("pm2_5", PM25_BREAKPOINTS)
    pm10_si = _build_sub_index_expr("pm10", PM10_BREAKPOINTS)
    so2_si = _build_sub_index_expr("so2", SO2_BREAKPOINTS)
    no2_si = _build_sub_index_expr("no2", NO2_BREAKPOINTS)

    # Add sub-index columns temporarily
    df = (
        df
        .withColumn("_si_pm25", pm25_si)
        .withColumn("_si_pm10", pm10_si)
        .withColumn("_si_so2", so2_si)
        .withColumn("_si_no2", no2_si)
    )

    # AQI = max of all sub-indices (F.greatest ignores nulls)
    df = df.withColumn(
        "AQI",
        F.greatest(
            F.col("_si_pm25"),
            F.col("_si_pm10"),
            F.col("_si_so2"),
            F.col("_si_no2")
        )
    )

    # Drop temporary sub-index columns
    df = df.drop("_si_pm25", "_si_pm10", "_si_so2", "_si_no2")

    print("[INFO] Computed AQI column using CPCB sub-index breakpoints (max rule) — pure SQL, no UDF")
    return df


def drop_missing_aqi(df):
    """
    Drop rows where AQI could not be computed (all pollutants were null).

    Args:
        df (DataFrame): Dataset with AQI column.

    Returns:
        DataFrame: Filtered dataset.
    """
    before = df.count()
    df = df.filter(F.col("AQI").isNotNull())
    after = df.count()
    print(f"[INFO] Dropped {before - after} rows with null AQI ({after} remaining)")
    return df


def impute_median(df, columns=None):
    """
    Impute missing values in pollutant columns with column-wise median.

    Args:
        df (DataFrame): Dataset.
        columns (list): Columns to impute. Defaults to pollutant columns.

    Returns:
        DataFrame: Dataset with imputed values.
    """
    if columns is None:
        columns = ["so2", "no2", "pm10", "pm2_5"]

    for col_name in columns:
        if col_name in df.columns:
            median_val = df.approxQuantile(col_name, [0.5], 0.01)
            if median_val and median_val[0] is not None:
                df = df.fillna({col_name: median_val[0]})
                print(f"[INFO] Imputed '{col_name}' missing values with median = {median_val[0]:.2f}")

    return df


def remove_outliers_iqr(df, column="AQI", factor=1.5):
    """
    Remove outliers from a column using the IQR method.

    Args:
        df (DataFrame): Dataset.
        column (str): Column to check for outliers.
        factor (float): IQR multiplier (default 1.5).

    Returns:
        DataFrame: Dataset with outliers removed.
    """
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    before = df.count()
    df = df.filter((F.col(column) >= lower_bound) & (F.col(column) <= upper_bound))
    after = df.count()

    print(f"[INFO] IQR outlier removal on '{column}': Q1={q1:.1f}, Q3={q3:.1f}, "
          f"IQR={iqr:.1f}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"[INFO] Removed {before - after} outlier rows ({after} remaining)")
    return df


def parse_date_features(df):
    """
    Convert the 'date' column to a proper timestamp and extract temporal features.

    Extracts: year, month, day

    Note: PySpark 4.x strict mode crashes on to_date("NA"), so we clean
    the date column first and use coalesce with multiple formats.

    Args:
        df (DataFrame): Dataset with 'date' column.

    Returns:
        DataFrame: Dataset with parsed date and temporal feature columns.
    """
    # Step 1: Clean NA/None/empty strings from the date column
    df = df.withColumn(
        "date",
        F.when(
            (F.trim(F.col("date")) == "") |
            (F.upper(F.trim(F.col("date"))) == "NA") |
            (F.upper(F.trim(F.col("date"))) == "NONE"),
            F.lit(None)
        ).otherwise(F.col("date"))
    )

    # Step 2: Parse date with multiple format fallbacks
    df = df.withColumn(
        "date_parsed",
        F.coalesce(
            F.to_date(F.col("date"), "yyyy-MM-dd"),
            F.to_date(F.col("date"), "dd-MM-yyyy"),
            F.to_date(F.col("date"), "MM/dd/yyyy"),
            F.to_date(F.col("date"), "yyyy/MM/dd")
        )
    )

    # Extract temporal features
    df = (
        df
        .withColumn("year", F.year("date_parsed"))
        .withColumn("month", F.month("date_parsed"))
        .withColumn("day", F.dayofmonth("date_parsed"))
    )

    # Drop rows where date could not be parsed
    before = df.count()
    df = df.filter(F.col("date_parsed").isNotNull())
    after = df.count()
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with unparseable dates")

    print("[INFO] Extracted temporal features: year, month, day")
    return df


def run_preprocessing(df):
    """
    Execute the full preprocessing pipeline.

    Pipeline:
      1. Rename rspm → pm10
      2. Impute missing pollutant values (median)
      3. Compute AQI from CPCB breakpoints
      4. Drop rows with null AQI
      5. Remove outliers (IQR on AQI)
      6. Parse dates and extract temporal features

    Args:
        df (DataFrame): Raw dataset.

    Returns:
        DataFrame: Cleaned and preprocessed dataset.
    """
    print("\n" + "=" * 60)
    print(" PREPROCESSING PIPELINE")
    print("=" * 60)

    df = rename_columns(df)
    df = cast_numeric_columns(df)
    df = impute_median(df, columns=["so2", "no2", "pm10", "pm2_5"])
    # Cache after imputation to materialize numeric columns.
    # This prevents the expression tree from growing beyond 64KB
    # (Spark codegen limit) when AQI sub-index chains are added.
    df.cache()
    df = compute_aqi_column(df)
    df = drop_missing_aqi(df)
    df = remove_outliers_iqr(df, column="AQI")
    df = parse_date_features(df)

    # Use 'location' as the city identifier for partitioning
    if "location" in df.columns:
        df = df.withColumnRenamed("location", "city")
        print("[INFO] Using 'location' column as 'city' for partitioning")

    print(f"[INFO] Preprocessing complete. Final shape: {df.count()} rows, {len(df.columns)} columns")
    print("=" * 60 + "\n")
    return df


if __name__ == "__main__":
    from spark_pipeline.data_ingestion import create_spark_session, load_data

    spark = create_spark_session()
    raw_df = load_data(spark, local_path="data/india_air_quality.csv")
    clean_df = run_preprocessing(raw_df)
    clean_df.show(10)
    clean_df.printSchema()
    spark.stop()
