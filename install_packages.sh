#!/usr/bin/env bash

pip install scikit-learn 

git clone https://github.com/nathanin/svsutils.git
cd svsutils
pip install -e .

pip install simple_gpu_scheduler
