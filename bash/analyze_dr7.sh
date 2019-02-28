#!/bin/bash
source activate py3p6

# codes
ablation=/Users/rezaie/github/SYSNet/src/ablation.py
multfit=/Users/rezaie/github/SYSNet/src/mult_fit.py
nnfit=/Users/rezaie/github/SYSNet/src/nn_fit.py
split=/Users/rezaie/github/SYSNet/src/add_features-split.py


# DATA
# output dirs & labels
glmp5=/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.5.r.npy
oudr_ab=/Volumes/TimeMachine/data/DR7/results/ablation/
oudr_r=/Volumes/TimeMachine/data/DR7/results/regression/
mult1=mult_all
mult2=mult_depz
log_ab=dr7.log
nn1=nn_ab

# MOCKS
pathmock=/Volumes/TimeMachine/data/mocks/3dbox/
umockext=*/*.hp.256.fits 
umock5l=.hp.256.5.r.npy
mockfeat=/Volumes/TimeMachine/data/mocks/mocks.DR7.table.fits
mlog_ab=mock.log


# ================ RUNS ====================
# DATA
# REGRESSION
#
# Feb 20: Ablation on DR7
mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab
# took 50 min

# Feb 21: Linear/quadratic multivariate fit on DR7 
#         Linear/quadratic depth-z fit on DR7
#         NN fit on DR7 with ablation
#
# python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split
# python $multfit --input $glmp5 --output ${oudr_r}${mult2}/ --split --ax 5
# took around 10 secs
#
mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab}.npy
# took 30 min on DR7


# ============= MOCKS =======================
# Feb 22
# Add features to uncontaminated
# mpirun --oversubscribe -np 4 python $split --hpmap $pathmock --ext $umockext --features $mockfeat --split r
# took 2 min

# Ablation on mocks
for i in $(seq -f "%03g" 1 100)
do
   mglmp5=${pathmock}${i}/${i}${umock5l}
   moudr_ab=${pathmock}${i}/results/ablation/
   echo ablation on $mglmp5
   mpirun --oversubscribe -np 5 python $ablation --data $mglmp5 --output ${moudr_ab} --log ${i}.$mlog_ab
done
# took 45 hours


# Feb 25
# Lin/quadratic fit on null mocks
# NN with ablation fit 
#
for i in $(seq -f "%03g" 1 100)
do
  mglmp5=${pathmock}${i}/${i}${umock5l}
  moudr_r=${pathmock}${i}/results/regression/
  moudr_ab=${pathmock}${i}/results/ablation/
  echo fit on $mglmp5
  #python $multfit --input $mglmp5 --output ${moudr_r}${mult1}/ --split
  mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn1}/ --ablog ${moudr_ab}${i}.${mlog_ab}.npy
done
# took 4 h
