#!/bin/bash

module purge
module load anaconda3/2025.6

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
