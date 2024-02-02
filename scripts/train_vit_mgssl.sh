#!/bin/bash

# Get the absolute path to the directory where the script is located
echo "Getting the script directory..."
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $script_dir"

# cd into the script's directory
cd "$script_dir"

train_py_path="$../training/train.py"

python3 "$train_py_path" \
                --model_arch vit_m2f \
                --use_proto_seg False \
                --frozen_target True \
                --mask_input True \
                $*