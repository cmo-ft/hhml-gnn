#!/bin/bash

# source ~/atlas.env
source ~/cvmfs.env

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="${SCRIPT_DIR}":"${PYTHONPATH}"
