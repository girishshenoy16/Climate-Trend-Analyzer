import subprocess
import sys
import os
import time


print("=" * 75)
print("CLIMATE TREND ANALYZER - FULL PIPELINE STARTED")
print("=" * 75)


# ======================================================
# SCRIPTS IN EXECUTION ORDER
# ======================================================

scripts = [
    "preprocess.py",
    "feature_engineering.py",
    "train_model.py",
    "evaluate_model.py",
    "forecast.py",
    "anomaly.py"
]


# ======================================================
# CURRENT DIRECTORY = src
# ======================================================

BASE_DIR = os.path.dirname(__file__)


# ======================================================
# RUN PIPELINE
# ======================================================

for script in scripts:

    path = os.path.join(BASE_DIR, script)

    print("\n" + "=" * 75)
    print(f"RUNNING: {script}")
    print("=" * 75)

    start = time.time()

    result = subprocess.run(
        [sys.executable, path]
    )

    end = time.time()

    if result.returncode != 0:
        print(f"\n❌ ERROR in {script}")
        print("Pipeline Stopped.")
        sys.exit(1)

    print(f"✅ Completed {script}")
    print(f"⏱ Time Taken: {round(end - start, 2)} sec")


print("\n" + "=" * 75)
print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 75)