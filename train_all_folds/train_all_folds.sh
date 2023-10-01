#!/bin/bash

cwd=`pwd`
model_dir=${cwd}/../gnn_template/

nfolds=3
for (( ifold=0; ifold<nfolds; ifold=ifold+1 )); do
    echo $ifold
    mkdir -p fold$ifold && cd fold$ifold
    cp $model_dir/run.sh ./
    sed -i "s|ifold=0|ifold=${ifold}|g" ./run.sh
    sed -i "s|# source ~/hailing.env|source ~/hailing.env|g" ./run.sh
    sed -i "s|main.py|${model_dir}/main.py|g" ./run.sh
    sed -i "s|log_dir=../|log_dir=./|g" ./run.sh
    # sed -i "s|apply_only=0|apply_only=1|g" ./run.sh
    # sed -i "s|multilepton/dnn/dataset/|multilepton/dnn/dataset/scripts|g" ./run.sh
    
    echo "    Universe   = vanilla
    Executable = run.sh
    Arguments  = 
    Log        = log
    Output     = out
    Error      = err
    request_GPUs =1 
    Queue" > gpujob.condor
    condor_submit gpujob.condor
    cd ..
done
