#!/bin/bash -x
. /root/.bashrc
pip install --upgrade pip
if [ "$(uname -i)" = "x86_64" ]; then
  if [ $DEVICE="xla" ]; then
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install "optimum[neuronx, diffusers]"
    python compile-sd2.py 
  fi
fi
tar -czvf /${COMPILER_WORKDIR_ROOT}/${MODEL_FILE}.tar.gz /${COMPILER_WORKDIR_ROOT}/
aws s3 cp /${COMPILER_WORKDIR_ROOT}/${MODEL_FILE}.tar.gz s3://${BUCKET}/${MODEL_FILE}_${DEVICE}_bsize_${BATCH_SIZE}.tar.gz
while true; do sleep 1000; done
