conda create -n frozen python=3.7
conda install opencv -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers tokenizers=0.10.1
conda install tqdm pandas scipy matplotlib psutil humanize scikit-learn
conda install av sacred  timm einops dominate -c conda-forge

pip install decord
