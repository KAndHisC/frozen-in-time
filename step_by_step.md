conda create -n frozen python=3.6

# GPU
conda install py-opencv opencv libopencv torchvision torchaudio pytorch=1.10.0 cudatoolkit=10.2 -c pytorch
# IPU
conda install py-opencv opencv libopencv torchvision torchaudio pytorch=1.10.0 cpuonly -c pytorch

conda install av sacred einops dominate -c conda-forge
conda install transformers tokenizers=0.10.1 -c huggingface
conda install tqdm pandas scipy matplotlib psutil humanize scikit-learn

pip install timm
pip install decord
pip install --upgrade torch==1.10.0 torchvision

mkdir $CONDA_PREFIX/etc/conda/
mkdir $CONDA_PREFIX/etc/conda/activate.d/
mkdir $CONDA_PREFIX/etc/conda/deactivate.d/

cat>$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh<<EOF
#!/bin/bash
export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/localdata/cn-customer-engineering/takiw/cachedir"
export TMPDIR="/localdata/cn-customer-engineering/takiw/tmp"
export POPART_LOG_LEVEL=WARN
export POPTORCH_LOG_LEVEL=WARN
export POPLAR_LOG_LEVEL=WARN
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports", "autoReport.streamAtEachRun":"false"}'
export PVTI_OPTIONS='{"enable":"true", "directory":"./sys_report"}'

source /home/tempscratch/takiw/sdk/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
source /home/tempscratch/takiw/sdk/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
EOF

cat>$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh<<EOF
#!/bin/bash
unset POPLAR_SDK_ENABLED
unset POPLAR_ENGINE_OPTIONS
unset PVTI_OPTIONS
EOF

conda install av   -c conda-forge
conda install sacred -c conda-forge
conda install einops  -c conda-forge
conda install dominate -c conda-forge
pip install timm

pip install --upgrade  torchvision


python train_ipu.py --config configs/msrvtt_4f_i21k_ipu.json

nohup python train.py --config configs/msrvtt_4f_i21k-1v100.json  > logs/train_msrvtt.log 2>&1 &
nohup python train.py --config configs/webvid2m-pt-i2-1v100.json  > logs/train_webvid.log 2>&1 &

nohup python download.py --part 0  > logs/download_webvid_val_1.log 2>&1 &
