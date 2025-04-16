# robot-learning-final

Final project repository for Cornell University's **CS 4756: Robot Learning**.

This project uses environments and expert policies from the [Metaworld](https://github.com/Farama-Foundation/Metaworld) benchmark to evaluate and experiment with various robotic manipulation tasks.

---

## Setup Instructions

Clone this repository **with submodules** to ensure Metaworld is included:

```bash
git clone --recurse-submodules [url]
cd robot-learning-final
```

Set up a virtual environment:

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
python example_test.py
```

This will launch a MuJoCo viewer and execute a scripted policy on five randomized goal positions.

---

## Dependencies

> Do **not** install Metaworld from PyPI. It is included as a Git submodule and pulled from source for compatibility and customization.
