#!/usr/bin/env bash
set -e

python3 -m jupyter nbconvert ./oof_price_bin.ipynb --to script
time python3 oof_price_bin.py
