#!/bin/bash -x
pip install --upgrade pip
if [ "$(uname -i)" = "x86_64" ]; then
  if [ "$DEVICE" == "xla" ]; then
    #git clone https://github.com/huggingface/optimum-neuron.git
    #cd optimum-neuron
    #pip install .
    #cd ..
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install "optimum[neuronx, diffusers]"
  elif [[ "$DEVICE" == "cuda" ]]; then
    pip install nvitop bitsandbytes accelerate protobuf --no-cache-dir transformers sentencepiece
  fi
  uvicorn run-llama:app --host=0.0.0.0
fi
