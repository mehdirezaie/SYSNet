#!/bin/bash
source activate py3p6

ablation=/Users/rezaie/github/SYSNet/src/ablation.py
glmp5=/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.5.r.npy
oudr_ab=/Volumes/TimeMachine/data/DR7/results/ablation/
log_ab=dr7.log


# Feb 20: Ablation on DR7
#mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab
# took 54 min

 
