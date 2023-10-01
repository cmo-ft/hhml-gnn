#! /bin/bash

# source ~/hailing.env

# Configure input data 
filePrefix="/lustre/collider/mocen/project/multilepton/dnn/dataset/"
fileList=( "${filePrefix}/" )
ifold=0

log_dir=../

num_epochs=50
num_slices_train=1
num_slices_test=1
lr=0.008
batch_size=2024
# Apply only
apply_only=0
apply_net=${log_dir}/net.pt
apply_file_list=()

# Pre-train
pre_train=0
pre_net="./net.pt"
pre_log="./train-result.json"


python main.py --fileList "${fileList[@]}" \
                --num_slices_train $num_slices_train --num_slices_test $num_slices_test\
                --apply_only $apply_only  --pre_train $pre_train  --pre_net $pre_net  --pre_log $pre_log --num_epochs=$num_epochs \
                --lr $lr --batch_size $batch_size --ifold $ifold  --apply_net $apply_net --apply_file_list ${apply_file_list[@]} --logDir $log_dir
