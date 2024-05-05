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
  uvicorn run-sd2:app --host=0.0.0.0
fi
