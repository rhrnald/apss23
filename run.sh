#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --exclusive                         \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./model $@
