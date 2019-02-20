#!/bin/bash

# ============================
# Feb 20
# make imaging maps for DR7
#mpirun --oversubscribe -np 3 python /Users/rezaie/github/SYSNet/src/make_sysmaps.py

# make a galaxy catalog from DR7 sweep files
mpirun --oversubscribe -np 4 python /Users/rezaie/github/SYSNet/src/select_targets.py

