import subprocess
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = f"plots/eval_results_{timestamp}.csv"
os.makedirs("plots", exist_ok=True)

modes = ["ppo", "bc+pnn", "ppo+pnn", "bc+ppo+pnn"]

for mode in modes:
    print(f"\n==== Running mode: {mode} ====\n")
    subprocess.run(
        [
            "python",
            "final/final.py",
            "--mode",
            mode,
            "--csv-path",
            csv_path,
            "--timestamp",
            timestamp,
        ],
        check=True,
    )

# Run only plotting after all subprocesses are done
subprocess.run(
    ["python", "final/final.py", "--csv-path", csv_path, "--timestamp", timestamp],
    check=True,
)
