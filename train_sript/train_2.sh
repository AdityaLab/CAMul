#!/bin/bash
for i in 6 7 8 9 10  
  do 
    python train_hosp.py -e 202144 -m deploy_week_44_$i -c True -d $i  —start_model hosp_deploy_week_43_$i
  done
