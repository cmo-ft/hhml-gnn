#!/bin/bash

# source_dir=/lustre/collider/mocen/project/multilepton/data/mc16a/346343/
source_dir=${1}
target_dir="/lustre/collider/mocen/project/multilepton/dnn/trex/apply_all/data_with_score/"
script=/lustre/collider/mocen/project/multilepton/dnn/trex/apply_all/merge_score/merge_score.py

sid=`basename $source_dir`
dataname=`dirname $source_dir`
dataname=`basename $dataname`

datadir=$dataname/$sid

if [ ! -f $source_dir/pt_data/dataset.pt ]; then exit; fi
echo ${source_dir}

mkdir -p $target_dir/$datadir && cd $target_dir/$datadir

python $script -i $source_dir -o $target_dir/$datadir

