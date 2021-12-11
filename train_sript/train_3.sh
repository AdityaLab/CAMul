#!/bin/bash
for i in 11 12 13 14 15 
  do 
    python train_hosp2.py -e 202148 -m deploy_week_48_$i -c True -d $i 
#     —start_model hosp_deploy_week_47_$i
  done
