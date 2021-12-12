#!/bin/bash
    
i=1
while [ $i -le 30 ]
    do
    python ./train_hosp2.py -e 202148 -m deploy_week_48 -c True -d $i 
    ((i++))
    done
echo All done
