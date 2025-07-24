# Vibronic Dynamics Resource Estimation

This repository provides resource estimation analysis for the simulation of vibronic Hamiltonians using the algorithm described in Narrative Document.

It contains a fully executable Jupyter [notebook](Resource_Estimation_Tutorial.ipynb) that leverages an *experimental* PennyLane development branch to reproduce all qubit and Toffoli counts reported in the paper.

Follow the installations below:

## Installations

```bash
# 1. clone the repo
git clone https://github.com/paarth7777/XPRIZE-RE.git
cd XPRIZE-RE

# 2. create (optional) virtual environment
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate

# 3. install experimental PennyLane branch
pip install git+https://github.com/PennyLaneAI/pennylane.git@add_resource_templates_core

# 4. install extra dependencies required by pennylane/labs
pip install -r requirements_labs.txt


