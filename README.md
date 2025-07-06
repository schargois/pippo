# PiPPO
## Progressive Imitation PPO: Accelerating Multi-Task Robot Learning Curricula via Behavior Cloning Warm Starting on Progressive Neural Networks

Sterling Chargois & Carter Larsen's final project repository for Cornell University's **CS 4756: Robot Learning**.

This project was made from scratch with the task of exploring some hypothesis with robot learning. The paper of our analysis can be found [here](https://drive.google.com/file/d/12MTTcVa6_Ab2dlIAFc5pk4wEIVL405mg/view?usp=drive_link) ([Slides](https://docs.google.com/presentation/d/1sQmqXRWeaORDbHCq5-XMdyDYY714FdZrJQhXGEuz_pM/edit?usp=sharing)).

This project uses environments and expert policies from the [Metaworld](https://github.com/Farama-Foundation/Metaworld) benchmark to evaluate and experiment with various robotic manipulation tasks.

---

## Setup Instructions

Clone this repository **with submodules** to ensure Metaworld is included:

```bash
git clone --recurse-submodules [url]
cd robot-learning-final
```

Set up a virtual environment:

Note: Using a venv may not work on all machines and requirements may be required to be installed outside of a virtual environement.

```bash
python -m venv .venv
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Example

To run the demonstration policy for the `pick-place-v2` task:

```bash
python final\example_test.py
```

This will launch a MuJoCo viewer and execute a scripted policy on five randomized goal positions.

---

## Running the paper's results

To run the analysis used in our paper, do the following:

```bash
python final\final_subprocesses.py
```

This will train on 4 modes and should take roughtly 24 hours to complete. The subprocesses module can be modified to have different modes to run on different machines, but the summary graphs will not be produced. This can be solved by using the corresponding .csv files.

---

## Dependencies

> Do **not** install Metaworld from PyPI. It is included as a Git submodule and pulled from source for compatibility and customization. This is due to version conflicts.
