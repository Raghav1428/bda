"""
main.py
=======
End-to-end orchestrator for the Scalable Cross-City AQI Forecasting system.

Execution flow:
  1. Load data into Spark (HDFS or local fallback)
  2. Preprocess (CPCB AQI computation, imputation, outlier removal, date parsing)
  3. Feature engineering (next-day target, lag, rolling, cyclic, interaction, assembly + scaling)
  4. Strict chronological train/test split (NO random splitting)
  5. Train distributed models (LinearRegression, RandomForest, GBT)
  6. PRIMARY EVALUATION — Temporal forecasting (RMSE, MAE, R²)
  7. Extract feature importance from Random Forest
  8. [OPTIONAL] Cross-city validation (leave-one-city-out)
  9. Train LSTM baseline (non-distributed)
  10. Save all results and print comparison table

SCIENTIFIC CORRECTIONS APPLIED
-------------------------------
1. TEMPORAL LEAKAGE FIX:
   - REMOVED random 80/20 split fallback (caused temporal leakage).
   - Implemented strict chronological splitting via chronological_split().
   - Train set = earliest years covering ~80% of rows.
   - No shuffling, no randomness in split.

2. DETERMINISTIC AQI RECONSTRUCTION FIX:
   - REMOVED same-day AQI prediction (was formula reconstruction).
   - Implemented NEXT-DAY AQI prediction: AQI_target = lead(AQI, 1).
   - Features at time t predict AQI at time t+1.
   - All features are backward-looking only.

EVALUATION DESIGN
-----------------
PRIMARY evaluation: Chronological temporal forecasting.
  - Tests the model's ability to predict FUTURE AQI from PAST data.
  - This is the standard benchmark for deployment-grade forecasting.

OPTIONAL experiment: Cross-city spatial generalization.
  - Tests prediction on UNSEEN CITIES (trained on all other cities).
  - This is harder than temporal forecasting because:
      (a) each city has unique pollution sources, geography, and climate;
      (b) the model must generalize across spatial domains, not just time;
      (c) cross-city performance is NOT used as the primary benchmark
          because real deployments retrain per-city with local data.
  - Enable via RUN_CROSS_CITY_VALIDATION = True.

Usage:
  python main.py
"""

import os
import sys
import time
import random
import numpy as np

# ── Reproducibility ──
random.seed(42)
np.random.seed(42)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Fix for Windows: PySpark workers look for "python3" which doesn't exist.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from spark_pipeline.data_ingestion import create_spark_session, load_data
from spark_pipeline.preprocessing import run_preprocessing
from spark_pipeline.feature_engineering import run_feature_engineering
from spark_pipeline.train_models import train_all_models, chronological_split
from spark_pipeline.evaluate_models import (
    evaluate_all_models,
    extract_feature_importance,
    save_metrics_to_csv,
    print_comparison_table,
)
from spark_pipeline.cross_city_validation import run_cross_city_validation
from deep_learning.lstm_baseline import train_lstm


# ============================================================================
# Configuration
# ============================================================================
HDFS_PATH = "hdfs://localhost:9000/user/aqi_project/data/india_air_quality.csv"
LOCAL_PATH = os.path.join(PROJECT_ROOT, "data", "india_air_quality.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_OUTPUT = os.path.join(RESULTS_DIR, "metrics_output.csv")
FEATURE_IMP_OUTPUT = os.path.join(RESULTS_DIR, "feature_importance.csv")
CROSS_CITY_METRICS_OUTPUT = os.path.join(RESULTS_DIR, "cross_city_metrics.csv")
CROSS_CITY_N = 3  # Number of cities for cross-city validation

# ---------------------------------------------------------------------------
# CROSS-CITY VALIDATION FLAG
# ---------------------------------------------------------------------------
# Set to True to run the exploratory cross-city spatial generalization test.
# Set to False (default) to run only the primary temporal forecasting evaluation.
#
# Why this is optional:
#   Cross-city validation evaluates DOMAIN GENERALIZATION — whether a model
#   trained on cities A, B, C can predict AQI for an unseen city D.  This is
#   an important research question but is NOT the primary deployment scenario.
#   In practice, forecasting systems are retrained with local station data,
#   making temporal (chronological) evaluation the relevant benchmark.
#
#   Cross-city next-day forecasting is inherently harder because:
#     - Each city has unique pollution sources (industrial vs. vehicular).
#     - Local geography (valleys trap pollutants, coastal cities disperse them).
#     - Climate differences affect atmospheric chemistry and dispersion.
#   These factors make cross-city R² lower, which reflects genuine spatial
#   heterogeneity — NOT model failure.
# ---------------------------------------------------------------------------
RUN_CROSS_CITY_VALIDATION = False


def main():
    """
    Execute the full AQI forecasting pipeline.
    """
    pipeline_start = time.time()

    print("\n" + "#" * 70)
    print("#")
    print("#  Scalable Cross-City Distributed Framework for")
    print("#  Short-Term AQI Forecasting Using Apache Spark")
    print("#  and Ensemble Learning")
    print("#")
    print("#  >>> NEXT-DAY FORECASTING MODE (no temporal leakage) <<<")
    print("#")
    print("#" * 70 + "\n")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Create Spark Session
    # ------------------------------------------------------------------
    print("[STEP 1/9] Creating Spark session...")
    spark = create_spark_session(app_name="AQI_Forecasting_Pipeline")
    spark.conf.set("spark.sql.shuffle.partitions", "200")

    try:
        # ------------------------------------------------------------------
        # Step 2: Load Data (HDFS with local fallback)
        # ------------------------------------------------------------------
        print("\n[STEP 2/9] Loading dataset...")
        raw_df = load_data(spark, hdfs_path=HDFS_PATH, local_path=LOCAL_PATH)

        # ------------------------------------------------------------------
        # Step 3: Preprocessing
        # ------------------------------------------------------------------
        print("\n[STEP 3/9] Running preprocessing pipeline...")
        clean_df = run_preprocessing(raw_df)

        # ------------------------------------------------------------------
        # Step 4: Feature Engineering (includes next-day target creation)
        # ------------------------------------------------------------------
        print("\n[STEP 4/9] Running feature engineering pipeline...")
        feat_df, feature_names = run_feature_engineering(clean_df, scale=True)

        total_rows = feat_df.count()
        print(f"[INFO] Feature-engineered dataset: {total_rows} rows")

        # ------------------------------------------------------------------
        # Step 5: STRICT Chronological Train/Test Split
        # ------------------------------------------------------------------
        print("\n[STEP 5/9] Splitting data chronologically and training models...")
        train_df, test_df = chronological_split(feat_df, train_ratio=0.8, min_test_years=2)

        trained_models = train_all_models(train_df)

        # ==================================================================
        # Step 6: PRIMARY TEMPORAL FORECASTING EVALUATION
        #
        # This is the MAIN result of the paper.  Models are trained on
        # historical data (earlier years) and evaluated on future data
        # (later years) they have never seen.  This directly measures
        # next-day AQI forecasting skill.
        # ==================================================================
        print("\n[STEP 6/9] PRIMARY TEMPORAL FORECASTING EVALUATION...")
        print("\n" + "=" * 70)
        print("  PRIMARY EVALUATION — TEMPORAL NEXT-DAY AQI FORECASTING")
        print("  Train: historical years | Test: future years (chrono split)")
        print("=" * 70)
        all_metrics = evaluate_all_models(trained_models, test_df)

        # ------------------------------------------------------------------
        # Step 7: Feature Importance (Random Forest)
        # ------------------------------------------------------------------
        print("\n[STEP 7/9] Extracting feature importance...")
        if "RandomForest" in trained_models:
            rf_model = trained_models["RandomForest"][0]
            extract_feature_importance(rf_model, feature_names, FEATURE_IMP_OUTPUT)
        else:
            print("[WARN] Random Forest model not available for feature importance.")

        # ------------------------------------------------------------------
        # Step 8: Cross-City Validation (OPTIONAL — exploratory experiment)
        #
        # Why this is separate from the primary evaluation:
        #   Cross-city validation tests SPATIAL domain generalization —
        #   whether a model trained on some cities can predict AQI for a
        #   city it has never seen.  This is an important research question
        #   but does NOT reflect real-world deployment, where models are
        #   retrained with local data.
        #
        #   Cross-city next-day forecasting is harder than temporal
        #   forecasting because the model must generalize across cities
        #   with different pollution profiles, geography, and climate.
        # ------------------------------------------------------------------
        if RUN_CROSS_CITY_VALIDATION:
            print("\n[STEP 8/9] EXPLORATORY SPATIAL GENERALIZATION TEST...")
            print("\n" + "=" * 70)
            print("  EXPLORATORY EXPERIMENT — CROSS-CITY SPATIAL GENERALIZATION")
            print("  Strategy: Leave-One-City-Out (train on N-1, test on 1)")
            print("  NOTE: This is NOT the primary benchmark.")
            print("=" * 70)
            cross_city_results = run_cross_city_validation(
                feat_df, n_cities=CROSS_CITY_N
            )

            # Save cross-city metrics to a SEPARATE file
            save_metrics_to_csv(
                [
                    {
                        "model": r.get("model", f"RF_cross_{r.get('city', '?')}"),
                        "RMSE": r["RMSE"],
                        "MAE": r["MAE"],
                        "R2": r["R2"],
                        "training_time_sec": r.get("training_time_sec", 0),
                    }
                    for r in cross_city_results
                ],
                CROSS_CITY_METRICS_OUTPUT,
            )
            print(f"[INFO] Cross-city metrics saved to: {CROSS_CITY_METRICS_OUTPUT}")
        else:
            print("\n[STEP 8/9] Cross-city validation SKIPPED (flag disabled).")
            print("  Set RUN_CROSS_CITY_VALIDATION = True in main.py to enable.")

        # ------------------------------------------------------------------
        # Step 9: LSTM Baseline
        # ------------------------------------------------------------------
        print("\n[STEP 9/9] Training LSTM baseline...")
        lstm_metrics = train_lstm(
            feat_df,
            feature_columns=feature_names,
            label_col="AQI_target",
            window_size=24,
            epochs=20,
            batch_size=32
        )
        all_metrics.append(lstm_metrics)

        # ------------------------------------------------------------------
        # Save Primary Results
        # ------------------------------------------------------------------
        save_metrics_to_csv(all_metrics, METRICS_OUTPUT)
        print_comparison_table(all_metrics)

        # Pipeline summary
        pipeline_elapsed = time.time() - pipeline_start
        print("\n" + "#" * 70)
        print(f"#  PIPELINE COMPLETE — Total time: {pipeline_elapsed:.2f}s")
        print(f"#  Primary metrics saved to: {METRICS_OUTPUT}")
        print(f"#  Feature importance saved to: {FEATURE_IMP_OUTPUT}")
        if RUN_CROSS_CITY_VALIDATION:
            print(f"#  Cross-city metrics saved to: {CROSS_CITY_METRICS_OUTPUT}")
        print("#" * 70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Clean up
        spark.stop()
        print("[INFO] Spark session stopped.")


if __name__ == "__main__":
    main()
