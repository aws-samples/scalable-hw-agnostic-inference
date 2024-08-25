#!/bin/bash -x
pip install --upgrade pip
if [ "$(uname -i)" = "x86_64" ]; then
  if [ "$DEVICE" == "xla" ]; then
    git clone https://github.com/huggingface/optimum-neuron.git
    cd optimum-neuron
    pip install .
    cd ..
  elif [[ "$DEVICE" == "cuda" ]]; then
    pip install nvitop bitsandbytes accelerate protobuf --no-cache-dir transformers sentencepiece
  fi
elif [ "$(uname -i)" = "aarch64" ]; then
  if [ "$DEVICE" == "cpu" ]; then
    echo "aarch64 cpu placeholder"
  fi
fi
uvicorn run-bert:app --host=0.0.0.0
