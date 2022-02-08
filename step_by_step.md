conda create -n frozen python=3.6
conda install av sacred einops dominate -c conda-forge
conda install opencv -c pytorch
# GPU
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# IPU
conda install torchvision torchaudio cpuonly -c pytorch

conda install -c huggingface transformers tokenizers=0.10.1
conda install tqdm pandas scipy matplotlib psutil humanize scikit-learn


pip install decord


cat>$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh<<EOF
#!/bin/bash
export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/localdata/takiw/cachedir"
export TMPDIR="/localdata/takiw/tmp"
export POPART_LOG_LEVEL=DEBUG
export POPTORCH_LOG_LEVEL=TRACE
export POPLAR_LOG_LEVEL=DEBUG
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports"}'

source /localdata/takiw/sdk/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
source /localdata/takiw/sdk/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
EOF

cat>$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh<<EOF
#!/bin/bash
unset POPLAR_SDK_ENABLED
EOF

conda install av   -c conda-forge
conda install sacred -c conda-forge
conda install einops  -c conda-forge
conda install dominate -c conda-forge
pip install timm
pip install --upgrade torch torchvision


