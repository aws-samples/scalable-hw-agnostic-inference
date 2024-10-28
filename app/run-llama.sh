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
elif [ "$DEVICE" == "cpu" ]; then
    python3 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 diffusers transformers accelerate protobuf --no-cache-dir transformers sentencepiece
# elif [ "$(uname -i)" = "aarch64" ]; then
#   if [ "$DEVICE" == "cpu" ]; then
#     python3 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 diffusers transformers accelerate protobuf --no-cache-dir transformers sentencepiece
#   fi
fi
uvicorn run-llama:app --host=0.0.0.0
