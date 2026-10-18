"""Run the whole pipeline end to end with one command:

    python run_all.py            # fast
    python run_all.py --tune     # adds a small hyper-parameter search
"""
import subprocess
import sys

STEPS = [
    ("1/7 Generate simulated data (skipped if data/raw/telco_churn.csv exists)", ["-m", "src.generate_data"]),
    ("2/7 Clean + split data", ["-m", "src.data_prep"]),
    ("3/7 Exploratory analysis", ["-m", "src.eda"]),
    ("4/7 Train + compare models", ["-m", "src.train"]),
    ("5/7 Evaluate on test set", ["-m", "src.evaluate"]),
    ("6/7 Error analysis", ["-m", "src.error_analysis"]),
    ("7/7 Model card + sample predictions", None),
]


def run(args):
    result = subprocess.run([sys.executable, *args])
    if result.returncode != 0:
        sys.exit(f"Step failed: {' '.join(args)}")


def main() -> None:
    extra = sys.argv[1:]
    for title, args in STEPS:
        print(f"\n===== {title} =====")
        if args is None:
            run(["-m", "src.model_card"])
            run(["-m", "src.predict", "--input", "data/processed/sample_new_customers.csv",
                 "--output", "reports/sample_predictions.csv"])
        elif args == ["-m", "src.train"]:
            run(args + extra)
        else:
            run(args)
    print("\nAll done. See reports/ for metrics, figures and the model card.")


if __name__ == "__main__":
    main()
