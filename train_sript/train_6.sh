#!/bin/bash
for i in 26 27 28 29 30 
  do 
    python train_hosp2.py -e 202148 -m deploy_week_48_$i -c True -d $i
#     —start_model hosp_deploy_week_43_$i
  done
