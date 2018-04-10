#!/bin/bash

set -e

for i in `seq 0 4`; do
  echo $i
  python ./train.py
done
