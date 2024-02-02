#!/bin/bash

# Get the absolute path to the directory where the script is located
dir_command="cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" &> /dev/null && pwd"
echo "Executing: $dir_command"
script_dir=$(eval $dir_command)

# Define the path to the train.py script relative to the script's directory
train_py_path="$script_dir/../training/train.py"

# Construct the Python script execution command
python_command="python3 \"$train_py_path\" --model_arch vit_m2f --use_proto_seg False --frozen_target True --mask_input True $*"

# Announce and run the Python script
echo "Executing: $python_command"
eval $python_command
