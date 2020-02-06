#!/bin/bash
eval "$(/Users/rezaie/anaconda3/bin/conda shell.bash hook)"
conda activate py3p6

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
oudr_rf=/Volumes/TimeMachine/data/DR7/results_referee/
maskc=/Volumes/TimeMachine/data/DR7/mask.cut.hp.256.fits    # remove pixels with extreme weights
maskdm=/Volumes/TimeMachine/data/DR7/mask_data_mock.cut.hp.256.fits
mult1=mult_all
mult2=mult_depz
mult3=mult_ab
mult4=mult_f
log_ab=dr7.log
nn1=nn_ab
nn3=nn_p
nn4=nn_f


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

# Jun 21: Ablation on DR7
# took 18 min
# for rk in 0 1 2 3 4
# do
#   time mpirun --oversubscribe -np 5 python $ablation --data $glmp5 --output $oudr_ab --log $log_ab --rank $rk
# done


# Jun 9: Linear/quadratic multivariate fit on DR7 
#         NN fit on DR7 with ablation
#         Run DR7 NN fit w/o ablation
# python $multfit --input $glmp5 --output ${oudr_r}${mult1}/ --split
# took around 20 secs
# ablation picks up [0, 1, 2, 7, 10, 11, 12, 14, 16, 17]
#mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn1}/ --ablog ${oudr_ab}${log_ab}
#mpirun --oversubscribe -np 5 python $nnfit --input $glmp5 --output ${oudr_r}${nn3}/
# took 75 min

# fit linear with validation
# mpirun --oversubscribe -np 5 python /Users/rezaie/github/SYSNet/src/mult_fit_vl.py --input $glmp5 --output /Volumes/TimeMachine/data/DR7/results/regression/mult_all_vl/ 
# took 15 secs on 5 processes

# run the weight footprint cut
# python /Users/rezaie/github/SYSNet/src/make_common_mask-data.py
# took 1 min
#
# Jun 10: Clustering after fixing the bug in C_ell, with new weights
# Run NNbar
# C_l for DR7 with and w/o ablation
# Run corr. functions
# each takes 20 min on 4 mpi process
# took 112 m
#for wname in uni lin quad
#do
#  wmap=${oudr_r}${mult1}/${wname}-weights.hp256.fits
#  mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --clfile cl_$wname --nnbar nnbar_$wname --corfile xi_$wname 
#done
#for nni in $nn1 $nn3
#do
# wmap=${oudr_r}${nni}/nn-weights.hp256.fits
# mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap $wmap --nnbar nnbar_$nni --clfile cl_$nni  --corfile xi_$nni 
#done
#mpirun --oversubscribe -np 2 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap /Volumes/TimeMachine/data/DR7/results/regression/mult_all_vl/lin-weights.hp256.fits --clfile cl_linvl 
#
# auto C_l for systematics
#mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskc --oudir $oudr_c --verbose --wmap none --clsys cl_sys --corsys xi_sys 


# Jan 4, 2019: run NNbar for the data on the mock footprint
# 20 minutes
#for wname in uni lin quad
#do
#  wmap=${oudr_r}${mult1}/${wname}-weights.hp256.fits
#  du -h $glmp $rnmp $drfeat $maskdm $wmap
#  mpirun -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskdm --oudir $oudr_rf --verbose --wmap $wmap --clfile cl_$wname --nnbar nnbar_$wname --corfile xi_$wname 
#done
#for nni in $nn1 $nn3
#do
# wmap=${oudr_r}${nni}/nn-weights.hp256.fits
# du -h $wmap
# mpirun -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskdm --oudir $oudr_rf --verbose --wmap $wmap --nnbar nnbar_$nni --clfile cl_$nni  --corfile xi_$nni 
#done
#mpirun -np 4 python $docl --galmap $glmp --ranmap $rnmp --photattrs $drfeat --mask $maskdm --oudir $oudr_rf --verbose --wmap none --clsys cl_sys --corsys xi_sys 
#


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
# took 2465 m for cont. and null mocks
#for i in $(seq -f "%03g" 1 100)
#do
#   mglmp5=${pathmock}${i}/${i}${umock5l}
#   moudr_ab=${pathmock}${i}/results/ablation/
#   echo "ablation on $mglmp5"
#   for rk in 0 1 2 3 4
#   do
#     mpirun --oversubscribe -np 5 python $ablation --data $mglmp5 --output ${moudr_ab} --log ${i}.$mlog_ab --rank $rk
#   done
#done


# Lin/quadratic fit on null mocks
# NN with ablation fit 
# June 27
# nn wo ablation
# nn & lin with few maps : 0 1 2 3 8
#for i in $(seq -f "%03g" 1 100)
#do
# mglmp5=${pathmock}${i}/${i}${umock5l}
# moudr_r=${pathmock}${i}/results/regression/
# moudr_ab=${pathmock}${i}/results/ablation/
# echo "fit on $mglmp5"
# python $multfit --input $mglmp5 --output ${moudr_r}${mult1}/ --split
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn1}/ --ablog ${moudr_ab}${i}.${mlog_ab}
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn3}/ 
# python $multfit --input $mglmp5 --output ${moudr_r}${mult4}/ --split --split --ax 0 1 2 3 8
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn4}/ --ax 0 1 2 3 8 
#done

# find common mask between cont. and null mocks
#python src/make_common_mask.py
#total number of weights : 1400
#real    2m32.280s
#user    7m5.730s
#sys     0m27.353s


#
# Clustering
# Use the median to upweight galaxies
# July 1: clustering after ablation and hp training for each fold
# 745 min
#for i in $(seq -f "%03g" 1 100)
#do
#   mglmp=${pathmock}${i}/${i}${umockl}
#   moudr_r=${pathmock}${i}/results/regression/
#   moudr_c=${pathmock}${i}/results/clustering/
   #   no weight - lin - weight
#   for multw in uni lin quad
#   do
#     wmap=${moudr_r}${mult1}/${multw}-weights.hp256.fits
#     clnm=cl_${multw}
#     nnnm=nnbar_${multw}
#     echo "clustering & nbar on $mglmp w $wmap"
#     mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0 
#   done
#   multw=$mult4
#   wmap=${moudr_r}${mult4}/lin-weights.hp256.fits
#   clnm=cl_${multw}
#   nnnm=nnbar_${multw}
#   echo "clustering & nbar on $mglmp w $wmap"
#   mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0 
#   # nn weights
#   for nni in $nn1 $nn3 $nn4
#    do
#     wmap=${moudr_r}${nni}/nn-weights.hp256.fits
#     clnm=cl_$nni
#     nnnm=nnbar_${nni}
#     echo "clustering & nbar on $mglmp w $wmap"
#     mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0 
#  done
#done

#
# Contaminated
#
#for i in $(seq -f "%03g" 1 100)
#do
#   mglmp5=${pathmock}${i}/$clab/${clab}_${i}${umock5l}
#   moudr_ab=${pathmock}${i}/$clab/results/ablation/
#   echo "ablation on $mglmp5"
#   for rk in 0 1 2 3 4
#   do 
#      mpirun --oversubscribe -np 5 python $ablation --data $mglmp5 --output ${moudr_ab} --log ${i}.$mlog_ab --rank $rk
#   done
#done
#
#for i in $(seq -f "%03g" 1 100)
#do
# mglmp5=${pathmock}${i}/$clab/${clab}_${i}${umock5l}
# moudr_r=${pathmock}${i}/$clab/results/regression/
# moudr_ab=${pathmock}${i}/$clab/results/ablation/
# echo "fit on $mglmp5"
# python $multfit --input $mglmp5 --output ${moudr_r}${mult1}/ --split
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn1}/ --ablog ${moudr_ab}${i}.${mlog_ab}
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn3}/ 
# python $multfit --input $mglmp5 --output ${moudr_r}${mult4}/ --split --ax 0 1 2 3 8
# mpirun --oversubscribe -np 5 python $nnfit --input $mglmp5 --output ${moudr_r}${nn4}/ --ax 0 1 2 3 8
#done


#
# we drop this for the paper
# fit the true contamination model on the contaminated data
#for i in $(seq -f "%03g" 1 100)
#do
# mglmp5=${pathmock}${i}/$clab/${clab}_${i}${umock5l}
## moudr_r=${pathmock}${i}/$clab/results/regression/
# echo "fit on $mglmp5"
# python $multfit --input $mglmp5 --output ${moudr_r}${mult3}/ --split --ax 0 1 2 7 10 11 12 14 16 17
## mglmp=${pathmock}${i}/$clab/${clab}_${i}${umockl}
# moudr_c=${pathmock}${i}/$clab/results/clustering/
# wmap=${moudr_r}${mult3}/lin-weights.hp256.fits
# clnm=cl_lin_ab
# echo "clustering on $mglmp w $wmap"
# mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose 
#done



# Clustering
#
# auto C_l for systematics
# galmap does not matter
#mpirun --oversubscribe -np 4 python $docl --galmap $glmp --ranmap $mfrac --photattrs $mockfeat --mask $mmaskcl --oudir ${pathmock}  --verbose --wmap none --clsys cl_sys


#
#for i in $(seq -f "%03g" 1 100)
#do
#  mglmp=${pathmock}${i}/$clab/${clab}_${i}${umockl}
#  moudr_r=${pathmock}${i}/$clab/results/regression/
#  moudr_c=${pathmock}${i}/$clab/results/clustering/
#  #
#  #   no weight - lin - weight
#  for multw in uni lin quad
#  do
#     wmap=${moudr_r}${mult1}/${multw}-weights.hp256.fits
#     clnm=cl_${multw}
#     nnnm=nnbar_${multw}
#     echo "clustering & nbar on $mglmp w $wmap"
#     mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0
#  done
#  # linear with few maps
# multw=$mult4
# wmap=${moudr_r}${mult4}/lin-weights.hp256.fits
# clnm=cl_${multw}
# nnnm=nnbar_${multw} 
# echo "clustering & nbar on $mglmp w $wmap"
# mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0
 # We do not do this for the paper
 # fit truth linear
 #wmap=${moudr_r}${mult3}/lin-weights.hp256.fits
 #clnm=cl_lin_ab
#  echo "clustering on $mglmp w $wmap"
#  mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose --njack 0
#   nnnm=nnbar_lin_ab
# echo "nnbar on $mglmp w $wmap"
# python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0
#
 #
 # linear truth
 #wmap=${oudr_r}${mult3}/lin-weights.hp256.fits
# clnm=cl_truth
# nnnm=nnbar_truth
#echo "clustering & nbar on $mglmp w $wmap"
#mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose --njack 0
# nn weights
# for nni in $nn1 $nn3 $nn4
#  do
#  wmap=${moudr_r}${nni}/nn-weights.hp256.fits
#  clnm=cl_$nni
#  nnnm=nnbar_${nni}
#  echo "clustering & nbar on $mglmp w $wmap"
#  mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --nnbar ${nnnm} --oudir ${moudr_c} --verbose  --njack 0
# done
#done

# one mock with Jackknife
#i=001
#mglmp=${pathmock}${i}/${i}${umockl}
#moudr_r=${pathmock}${i}/results/regression/
#moudr_c=${pathmock}${i}/results/clustering/
#multw=uni_wjack
#wmap=${moudr_r}${mult1}/${multw}-weights.hp256.fits
#clnm=cl_${multw}
#echo "clustering on $mglmp w $wmap"
#mpirun --oversubscribe -np 4 python $docl --galmap ${mglmp} --ranmap ${mfrac} --photattrs ${mockfeat} --wmap $wmap --mask ${mmaskcl} --clfile ${clnm} --oudir ${moudr_c} --verbose --njack 20


