#!/bin/bash
for i in 6 7 8 9 10  
  do 
    python train_hosp.py -e 202148 -m deploy_week_48_$i -c True -d $i 
#     —start_model hosp_deploy_week_47_$i
  done
