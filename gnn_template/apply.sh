#! /bin/bash

ifold=${1}

template_dir="/lustre/collider/mocen/project/multilepton/hhml-gnn/gnn_template"

batch_size=20240
# Apply only
apply_only=1
datadir=/lustre/collider/mocen/project/multilepton/data/
apply_file_list=`ls ${datadir}/*/*/*.pt`

net_dir="/lustre/collider/mocen/project/multilepton/hhml-gnn/train_all_folds/"
nfolds=3

apply_net=${net_dir}/fold${ifold}/net.pt

python ${template_dir}/main.py  --apply_only $apply_only --ifold $ifold  --apply_net $apply_net --apply_file_list ${apply_file_list[@]} 
