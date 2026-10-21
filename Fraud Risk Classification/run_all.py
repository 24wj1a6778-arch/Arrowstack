"""Run the whole pipeline end to end with one command:

    python run_all.py
"""
import subprocess
import sys

STEPS = [
    ("1/9 Generate simulated transactions (skipped if data/raw/transactions.csv exists)", ["-m", "src.generate_data"]),
    ("2/9 Clean + split data", ["-m", "src.data_prep"]),
    ("3/9 Exploratory analysis", ["-m", "src.eda"]),
    ("4/9 Train + compare models and imbalance strategies", ["-m", "src.train"]),
    ("5/9 Threshold analysis (validation)", ["-m", "src.threshold_analysis"]),
    ("6/9 Evaluate on test set", ["-m", "src.evaluate"]),
    ("7/9 Explainability", ["-m", "src.explain"]),
    ("8/9 Error analysis", ["-m", "src.error_analysis"]),
    ("9/9 Model card + sample predictions", None),
]


def run(args):
    if subprocess.run([sys.executable, *args]).returncode != 0:
        sys.exit(f"Step failed: {' '.join(args)}")


def main() -> None:
    for title, args in STEPS:
        print(f"\n===== {title} =====")
        if args is None:
            run(["-m", "src.model_card"])
            run(["-m", "src.predict", "--input", "data/processed/sample_new_transactions.csv",
                 "--output", "reports/sample_predictions.csv"])
        else:
            run(args)
    print("\nAll done. See reports/ for metrics, figures and the model card.")


if __name__ == "__main__":
    main()
