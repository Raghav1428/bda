"""
data_ingestion.py
=================
Loads the India Air Quality dataset into a Spark DataFrame.
Supports reading from HDFS or falling back to local file path.
"""

from pyspark.sql import SparkSession


def create_spark_session(app_name="AQI_Forecasting"):
    """
    Create and return a SparkSession configured for local execution.

    Args:
        app_name (str): Name of the Spark application.

    Returns:
        SparkSession: Configured Spark session.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, hdfs_path=None, local_path=None):
    """
    Load CSV dataset from HDFS first; fall back to local path if unavailable.

    Args:
        spark (SparkSession): Active Spark session.
        hdfs_path (str): HDFS URI, e.g. "hdfs://localhost:9000/user/aqi_project/data/india_air_quality.csv"
        local_path (str): Local filesystem path to the CSV file.

    Returns:
        pyspark.sql.DataFrame: Raw dataset.
    """
    path = None

    # Try HDFS first
    if hdfs_path:
        try:
            df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
            print(f"[INFO] Loaded data from HDFS: {hdfs_path}")
            print(f"[INFO] Total rows: {df.count()}, Columns: {len(df.columns)}")
            return df
        except Exception as e:
            print(f"[WARN] HDFS read failed ({e}). Falling back to local path.")

    # Fallback to local
    if local_path:
        path = local_path
    else:
        raise FileNotFoundError("No valid data path provided (HDFS or local).")

    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"[INFO] Loaded data from local path: {path}")
    print(f"[INFO] Total rows: {df.count()}, Columns: {len(df.columns)}")
    return df


if __name__ == "__main__":
    spark = create_spark_session()
    df = load_data(
        spark,
        hdfs_path="hdfs://localhost:9000/user/aqi_project/data/india_air_quality.csv",
        local_path="data/india_air_quality.csv"
    )
    df.printSchema()
    df.show(5)
    spark.stop()
