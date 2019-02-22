#!/bin/bash
source activate py3p6

# codes
ablation=/Users/rezaie/github/SYSNet/src/ablation.py
multfit=/Users/rezaie/github/SYSNet/src/mult_fit.py
nnfit=/Users/rezaie/github/SYSNet/src/nn_fit.py

# output dirs & labels
glmp5=/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.5.r.npy
oudr_ab=/Volumes/TimeMachine/data/DR7/results/ablation/
oudr_r=/Volumes/TimeMachine/data/DR7/results/regression/
mult1=mult_all
mult2=mult_depz
log_ab=dr7.log
nn1=nn_ab

# ================ RUNS ====================
# REGRESSION
#
# Feb 20: Ablation on DR7
# mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab
# took 50 min

# Feb 21: Linear/quadratic multivariate fit on DR7 
#         Linear/quadratic depth-z fit on DR7
#         NN fit on DR7 with ablation
#
# python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split
# python $multfit --input $glmp5 --output ${oudr_r}${mult2}/ --split --ax 5
# took around 10 secs
#
# mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab}.npy
# took 30 min on DR7

# CLUSTERING
#
