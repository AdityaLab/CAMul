#!/bin/bash
for i in 16 17 18 19 20 
  do 
    python "train_hosp.py -e 202142 -m deploy_week_42_"$i" -c True -d "$i" --start_model hosp_deploy_week_41_"$i
  done
