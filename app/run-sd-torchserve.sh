#!/bin/bash -x

pip install --upgrade pip
pip install environment_kernels
pip install diffusers transformers accelerate safetensors matplotlib Pillow ipython torch -U
pip install click nvitop
while true; do sleep 1000; done
