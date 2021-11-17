#!/bin/bash
module load anaconda/2020.11
source activate py38
python train.py --gpu 0-1 --continue

