#!/bin/bash
module --force purge
PATH=${PATH}:/opt/software/singularity-3.0/bin/
singularity shell --nv --bind ${RAP},${HOME} /rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg

