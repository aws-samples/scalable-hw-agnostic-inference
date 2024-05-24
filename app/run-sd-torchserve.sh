#!/bin/bash -x

pip install --upgrade pip
if [ "$(uname -i)" = "x86_64" ]; then
  if [ "$DEVICE" == "xla" ]; then
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install "optimum[neuronx, diffusers]"
  elif [ "$DEVICE" == "cuda" ]; then
    pip install environment_kernels
    pip install diffusers transformers accelerate safetensors matplotlib Pillow ipython torch -U
    pip install click nvitop
  fi
  mkdir model-store
  torch-model-archiver --model-name stable-diffusion --version 2.1 --handler run-sd-torchserve.py
  mv stable-diffusion.mar /model-store
  torchserve --start --foreground --ts-config config.properties
fi
while true; do sleep 10000; done
