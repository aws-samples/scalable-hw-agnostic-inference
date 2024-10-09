#!/bin/bash -x
pip install --upgrade pip
#upgrading pytorch; not sure why its not in the DLC
#python -c "import torch; print(torch.__version__)"
#pip install --upgrade torch torchvision
#python -c "import torch; print(torch.__version__)"
if [ "$(uname -i)" = "x86_64" ]; then
  if [ "$DEVICE" == "xla" ]; then
    # TODO ideally that will be moved to Dockerfile but keeping it here for now. 
    python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    python -m pip install --upgrade-strategy eager optimum[neuronx]
  elif [[ "$DEVICE" == "cuda" ]]; then
    pip install nvitop bitsandbytes accelerate protobuf --no-cache-dir transformers sentencepiece
  fi
elif [ "$(uname -i)" = "aarch64" ]; then
  if [ "$DEVICE" == "cpu" ]; then
    python3 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 diffusers transformers accelerate
  fi
fi
uvicorn run-yolo:app --host=0.0.0.0
