#!/bin/bash
for i in 1 2 3 4 5  
  do 
    python train_hosp.py -e 202143 -m deploy_week_43_$i -c True -d $i  —start_model hosp_deploy_week_42_$i
  done

