#! /bin/bash

# source ~/hailing.env

# Configure input data 
filePrefix="/lustre/collider/mocen/project/multilepton/dnn/dataset/"
fileList=( "${filePrefix}/" )
ifold=0

num_epochs=50
num_slices_train=1
num_slices_test=1
num_slices_apply=1
lr=0.008
batch_size=2024
# Apply only
apply_only=0

# Pre-train
pre_train=0
pre_net="./net.pt"
pre_log="./train-result.json"


python gnn/main.py --fileList "${fileList[@]}" \
                --num_slices_train $num_slices_train --num_slices_test $num_slices_test --num_slices_apply $num_slices_apply\
                --apply_only $apply_only  --pre_train $pre_train  --pre_net $pre_net  --pre_log $pre_log --num_epochs=$num_epochs \
                --lr $lr --batch_size $batch_size --ifold $ifold 
