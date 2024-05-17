#!/bin/bash -x

pip install --upgrade pip
pip install environment_kernels
pip install diffusers transformers accelerate safetensors matplotlib Pillow ipython torch -U
pip install click nvitop
mkdir model-store
torch-model-archiver --model-name stable-diffusion --version 1.0 --handler run-sd-torchserve.py
mv stable-diffusion.mar /model-store
torchserve --start --ts-config config.properties
while true; do sleep 1000; done
