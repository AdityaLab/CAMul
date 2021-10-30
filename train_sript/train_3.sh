#!/bin/bash
for i in 11 12 13 14 15 
  do 
  	python "train_hosp.py -e 202142 -m deploy_week_42_"$i" -c True -d "$i" --start_model hosp_deploy_week_41_"$i
  done
