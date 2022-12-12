#!/bin/bash

WORKDIR=
EXTERNDIR=
PYTHONPATH="/home/yando/Workspace/CAI/"
LD_LIBRARY_PATH=
CUDA_VISIBLE_DEVICES=1

EXPORTS="PYTHONPATH=$PYTHONPATH"
EXPORTS="$EXPORTS LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
EXPORTS="$EXPORTS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

. $HOME/Workspace/CAI/.venv/aidev-3.7/bin/activate && export $EXPORTS && python main.py 
