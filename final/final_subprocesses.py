import subprocess

modes = ["ppo", "bc+pnn", "ppo+pnn", "bc+ppo+pnn"]

for mode in modes:
    print(f"\n==== Running mode: {mode} ====\n")
    subprocess.run(["python", "final/final.py", "--mode", mode], check=True)
