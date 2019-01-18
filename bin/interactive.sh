#!/bin/bash
# feature=k80
msub -N development -A colosse-users -l advres=MILA2019,nodes=1:gpus=1,walltime=15:00 -I

