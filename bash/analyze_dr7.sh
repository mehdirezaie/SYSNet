#!/bin/bash
source activate py3p6

# codes
ablation=/Users/rezaie/github/SYSNet/src/ablation.py
multfit=/Users/rezaie/github/SYSNet/src/mult_fit.py
nnfit=/Users/rezaie/github/SYSNet/src/nn_fit.py
split=/Users/rezaie/github/SYSNet/src/add_features-split.py
docl=/Users/rezaie/github/SYSNet/src/run_pipeline.py

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
umockl=.hp.256.fits
umock5l=.hp.256.5.r.npy
mockfeat=/Volumes/TimeMachine/data/mocks/mocks.DR7.table.fits
mlog_ab=mock.log
mmask=/Volumes/TimeMachine/data/mocks/mask.hp.256.fits
mfrac=/Volumes/TimeMachine/data/mocks/fracgood.hp256.fits

# ================ RUNS ====================
# DATA
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


# ============= MOCKS =======================
# Feb 22
# Add features to uncontaminated
# mpirun --oversubscribe -np 4 python $split --hpmap $pathmock --ext $umockext --features $mockfeat --split r
# took 2 min

# Ablation on mocks
#for i in $(seq -f "%03g" 1 100)
#do
#   mglmp5=${pathmock}${i}/${i}${umock5l}
#   moudr_ab=${pathmock}${i}/results/ablation/
#   echo ablation on $mglmp5
#   mpirun --oversubscribe -np 5 python $ablation --data $mglmp5 --output ${moudr_ab} --log ${i}.$mlog_ab
#done
# took 45 hours


# Feb 25
# Lin/quadratic fit on null mocks
# NN with ablation fit 
#
#for i in $(seq -f "%03g" 1 100)
#do
#  mglmp5=${pathmock}${i}/${i}${umock5l}
#  moudr_r=${pathmock}${i}/results/regression/
#  moudr_ab=${pathmock}${i}/results/ablation/
#  echo fit on $mglmp5
#  #python $multfit --input $mglmp5 --output ${moudr_r}${mult1}/ --split
#  mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn1}/ --ablog ${moudr_ab}${i}.${mlog_ab}.npy
#done
# took 4 h
#
# Clustering
# March 1
# Use the median to upweight galaxies
for i in $(seq -f "%03g" 1 100)
do
  mglmp=${pathmock}${i}/${i}${umockl}
  moudr_r=${pathmock}${i}/results/regression/
  moudr_c=${pathmock}${i}/results/clustering-upw/
  # no weight - lin - weight
  for multw in uni lin quad
  do
     wmap=${moudr_r}${mult1}/${multw}-weights.hp256.fits
     clnm=cl_${multw}
     echo "clustering on $mglmp w $wmap"
     mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmask} --clfile ${clnm} --oudir ${moudr_c} --verbose 
  done
 # nn weights
 wmap=${moudr_r}${nn1}/nn-weights.hp256.fits
 clnm=cl_nn
 echo "clustering on $mglmp w $wmap"
 mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmask} --clfile ${clnm} --oudir ${moudr_c} --verbose 
done

#
# use the median
#for i in 001 006 007 008 010 013 015 016 017 018 019 021 025 026 027 030 031 032 035 036 039 047 049 051 054 055 056 058 059 063 067 068 070 072 079 081 085 087 089 090 092
#do 
  #mglmp=${pathmock}${i}/${i}${umockl}
  #moudr_r=${pathmock}${i}/results/regression/
  #moudr_c=${pathmock}${i}/results/clustering/
 # nn weights
 #wmap=${moudr_r}${nn1}/nnm-weights.hp256.fits
 #clnm=cl_nnm
 #echo "clustering on $mglmp w $wmap"
 #mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmask} --clfile ${clnm} --oudir ${moudr_c} --verbose 
#done


