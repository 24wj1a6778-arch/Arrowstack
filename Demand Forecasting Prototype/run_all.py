"""Run the whole pipeline end to end with one command:

    python run_all.py
"""
import subprocess
import sys

STEPS = [
    ("1/9 Generate simulated sales (skipped if data/raw/sales.csv exists)", ["-m", "src.generate_data"]),
    ("2/9 Clean + time-ordered split", ["-m", "src.data_prep"]),
    ("3/9 Exploratory analysis (training period only)", ["-m", "src.eda"]),
    ("4/9 Compare models on validation (walk-forward backtest)", ["-m", "src.train"]),
    ("5/9 Evaluate on the test period", ["-m", "src.evaluate"]),
    ("6/9 Inventory simulation", ["-m", "src.inventory_sim"]),
    ("7/9 Error analysis", ["-m", "src.error_analysis"]),
    ("8/9 Model card", ["-m", "src.model_card"]),
    ("9/9 Forecast + order recommendations (demo)",
     ["-m", "src.predict", "--inventory-file", "data/processed/sample_inventory.csv",
      "--promo-file", "data/processed/sample_future_promos.csv"]),
]


def main() -> None:
    for title, args in STEPS:
        print(f"\n===== {title} =====")
        if subprocess.run([sys.executable, *args]).returncode != 0:
            sys.exit(f"Step failed: {' '.join(args)}")
    print("\nAll done. See reports/ for metrics, figures and the model card.")


if __name__ == "__main__":
    main()
