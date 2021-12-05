#!/bin/bash
for i in 1 2 3 4 5  
  do 
    python train_hosp.py -e 202148 -m deploy_week_48_$i -c True -d $i  —start_model hosp_deploy_week_47_$i
  done

