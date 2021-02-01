#!/bin/sh

squeue -u $USER | grep $USER | awk '{system( "head slurm-"$1".out -n1 && tail slurm-"$1".out -n5")}'
