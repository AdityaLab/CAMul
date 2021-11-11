#!/bin/bash
for i in 11 12 13 14 15 
  do 
    python train_hosp.py -e 202144 -m deploy_week_44_$i -c True -d $i  —start_model hosp_deploy_week_43_$i
  done
