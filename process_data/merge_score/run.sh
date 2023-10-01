#!/bin/bash

# source_data_pt=/lustre/collider/mocen/project/multilepton/data/mc16a/304014/user.sparajul.31752835._000002.output_output.pt
source_data_pt=${1}
target_dir=${2}
script=/lustre/collider/mocen/project/multilepton/hhml-gnn/process_data/merge_score/merge_score.py

source_dir=`dirname ${source_data_pt}`
sid=`basename $source_dir`
tmp=`dirname $source_dir`
dataname=`basename $tmp`

datadir=$dataname/$sid

# if [ ! -f $source_dir/pt_data/dataset.pt ]; then exit; fi
echo ${source_dir}

mkdir -p $target_dir/$datadir && cd $target_dir/$datadir

# python $script -i $source_data_pt -o $target_dir/$datadir/
python $script -i $source_data_pt -o ./

