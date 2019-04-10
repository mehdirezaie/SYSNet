#!/bin/bash
source activate py3p6

# codes
ablation=/Users/rezaie/github/SYSNet/src/ablation.py
multfit=/Users/rezaie/github/SYSNet/src/mult_fit.py
nnfit=/Users/rezaie/github/SYSNet/src/nn_fit.py
split=/Users/rezaie/github/SYSNet/src/add_features-split.py
docl=/Users/rezaie/github/SYSNet/src/run_pipeline.py
docont=/Users/rezaie/github/SYSNet/src/contaminate.py

# DATA
# output dirs & labels
glmp=/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.cut.hp256.fits
glmp5=/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.5.r.npy
drfeat=/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.fits
rnmp=/Volumes/TimeMachine/data/DR7/frac.hp.256.fits
oudr_ab=/Volumes/TimeMachine/data/DR7/results/ablation/
oudr_r=/Volumes/TimeMachine/data/DR7/results/regression/
oudr_c=/Volumes/TimeMachine/data/DR7/results/clustering/
maskc=/Volumes/TimeMachine/data/DR7/mask.cut.hp.256.fits    # remove pixels with extreme weights
mult1=mult_all
mult2=mult_depz
mult3=mult_ab
log_ab=dr7.log
nn1=nn_ab

# MOCKS
pathmock=/Volumes/TimeMachine/data/mocks/3dbox/
umockext=*/*.hp.256.fits 
cmockext=*/cp2p/cp2p*.hp.256.fits 
umockl=.hp.256.fits
umock5l=.hp.256.5.r.npy
mockfeat=/Volumes/TimeMachine/data/mocks/mocks.DR7.table.fits
mlog_ab=mock.log
mmask=/Volumes/TimeMachine/data/mocks/mask.hp.256.fits
mmaskc=/Volumes/TimeMachine/data/mocks/mask.cut.hp.256.fits
mmaskcl=/Volumes/TimeMachine/data/mocks/mask.cut.w.hp.256.fits
mfrac=/Volumes/TimeMachine/data/mocks/fracgood.hp256.fits
clab=cp2p


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

# March 6: 
# Run NNbar
# C_l for DR7
# for wname in uni lin quad
# do
#    wmap=${oudr_r}${mult1}/${wname}-weights.hp256.fits
#    mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --clfile cl_$wname --nnbar nnbar_$wname
# done
# wmap=${oudr_r}${nn1}/nn-weights.hp256.fits
# mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_$nn1 --clfile cl_$nn1 

# April 5:
# Run NNbar with equal area binning
#for wname in uni lin quad
#do
#    wmap=${oudr_r}${mult1}/${wname}-weights.hp256.fits
#    mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_$wname
#done
#wmap=${oudr_r}${nn1}/nn-weights.hp256.fits
#mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_$nn1 




# auto C_l for systematics
# mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --clsys cl_sys

# March 7: Run corr. functions
# each takes 10 min on 2 mpi process
# for wname in uni lin quad
# do
#    wmap=${oudr_r}${mult1}/${wname}-weights.hp256.fits
#    time mpirun --oversubscribe -np 2 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --corfile xi_$wname 
# done
#wmap=${oudr_r}${nn1}/nn-weights.hp256.fits
#time mpirun --oversubscribe -np 2 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --corfile xi_$nn1 
# auto corr. for systematics
# mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --corsys xi_sys





# ============= MOCKS =======================
# March 8: 
# ablation & regression took 4 days
# clustering for both clean and corrupted mocks
# took 150 min 
# Add features to uncontaminated with the removed mask
# mpirun --oversubscribe -np 4 python $split --hpmap $pathmock --ext $umockext --features $mockfeat --split r --mask $mmaskc
# took 2 min

# fit 10 maps to DR7
# time python $multfit --input $glmp5 --output ${oudr_r}${mult3}/ --split --ax 0 1 2 7 10 11 12 14 16 17
# contaminate mocks with 10 maps
# time python $docont $mockfeat ${oudr_r}${mult3}/regression_log.npy /Volumes/TimeMachine/data/mocks/3dbox/*/*.hp.256.fits
# Find the over-lapped negative masks, and remove them from the fiducial mock footprint
# add and split contaminated mocks
# mpirun --oversubscribe -np 4 python $split --hpmap $pathmock --ext $cmockext --features $mockfeat --split r --mask $mmaskc



# Ablation on mocks
#for i in $(seq -f "%03g" 1 100)
#do
#   mglmp5=${pathmock}${i}/${i}${umock5l}
#   moudr_ab=${pathmock}${i}/results/ablation/
#   echo "ablation on $mglmp5"
#   mpirun --oversubscribe -np 5 python $ablation --data $mglmp5 --output ${moudr_ab} --log ${i}.$mlog_ab
#done
# took 45 hours


# Lin/quadratic fit on null mocks
# NN with ablation fit 
#
#for i in $(seq -f "%03g" 1 100)
#do
# mglmp5=${pathmock}${i}/${i}${umock5l}
# moudr_r=${pathmock}${i}/results/regression/
# moudr_ab=${pathmock}${i}/results/ablation/
# echo "fit on $mglmp5"
# python $multfit --input $mglmp5 --output ${moudr_r}${mult1}/ --split
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn1}/ --ablog ${moudr_ab}${i}.${mlog_ab}.npy
#done
# took 4 h
#
# Clustering
# Use the median to upweight galaxies
#for i in $(seq -f "%03g" 1 100)
#do
#  mglmp=${pathmock}${i}/${i}${umockl}
#  moudr_r=${pathmock}${i}/results/regression/
#  moudr_c=${pathmock}${i}/results/clustering/
#  # no weight - lin - weight
#  for multw in uni lin quad
#  do
#     wmap=${moudr_r}${mult1}/${multw}-weights.hp256.fits
#     clnm=cl_${multw}
#     echo "clustering on $mglmp w $wmap"
#     mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose 
#  done
 # nn weights
# wmap=${moudr_r}${nn1}/nn-weights.hp256.fits
# clnm=cl_nn
# echo "clustering on $mglmp w $wmap"
# mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose 
#done


#
# Contaminated

#for i in $(seq -f "%03g" 1 100)
#do
#   mglmp5=${pathmock}${i}/$clab/${clab}_${i}${umock5l}
#   moudr_ab=${pathmock}${i}/$clab/results/ablation/
#   echo "ablation on $mglmp5"
#   mpirun --oversubscribe -np 5 python $ablation --data $mglmp5 --output ${moudr_ab} --log ${i}.$mlog_ab
#done

#for i in $(seq -f "%03g" 1 100)
#do
# mglmp5=${pathmock}${i}/$clab/${clab}_${i}${umock5l}
# moudr_r=${pathmock}${i}/$clab/results/regression/
# moudr_ab=${pathmock}${i}/$clab/results/ablation/
# echo "fit on $mglmp5"
# python $multfit --input $mglmp5 --output ${moudr_r}${mult1}/ --split
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn1}/ --ablog ${moudr_ab}${i}.${mlog_ab}.npy
#done



# Clustering
# Use the median to upweight galaxies
#for i in $(seq -f "%03g" 1 100)
#do
#  mglmp=${pathmock}${i}/$clab/${clab}_${i}${umockl}
#  moudr_r=${pathmock}${i}/$clab/results/regression/
#  moudr_c=${pathmock}${i}/$clab/results/clustering/
  # no weight - lin - weight
#  for multw in uni lin quad
#  do
#     wmap=${moudr_r}${mult1}/${multw}-weights.hp256.fits
#     clnm=cl_${multw}
#     echo "clustering on $mglmp w $wmap"
#     mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose 
#  done
# # nn weights
# wmap=${moudr_r}${nn1}/nn-weights.hp256.fits
# clnm=cl_nn
# echo "clustering on $mglmp w $wmap"
# mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose 
#done
#
# auto C_l for systematics
# galmap does not matter
# mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $mfrac --photattrs $mockfeat --mask $mmaskcl --oudir ${pathmock}  --verbose --wmap none --clsys cl_sys


# fit the true contamination model on the contaminated data
for i in $(seq -f "%03g" 1 100)
do
 mglmp5=${pathmock}${i}/$clab/${clab}_${i}${umock5l}
 moudr_r=${pathmock}${i}/$clab/results/regression/
 echo "fit on $mglmp5"
 python $multfit --input $mglmp5 --output ${moudr_r}${mult3}/ --split --ax 0 1 2 7 10 11 12 14 16 17
 mglmp=${pathmock}${i}/$clab/${clab}_${i}${umockl}
 moudr_c=${pathmock}${i}/$clab/results/clustering/
 wmap=${moudr_r}${mult3}/lin-weights.hp256.fits
 clnm=cl_lin_ab
 echo "clustering on $mglmp w $wmap"
 mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose 
done


