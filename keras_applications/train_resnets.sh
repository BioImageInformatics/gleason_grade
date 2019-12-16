#!/usr/bin/env bash

for j in $( ls ../data/patients*.txt ); do 
  echo python ./train_application.py $( realpath ../data/jpg_nameless/ ) \
    $( realpath ../data/mask_nameless/ ) \
    $( realpath $j ) \
    ./ResNet50V2_$(basename $j).h5
done