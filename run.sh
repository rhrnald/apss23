#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --exclusive                         \
  ./model $@
