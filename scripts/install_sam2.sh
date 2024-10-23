apt-get update && apt-get install -y --no-install-recommends ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev pkg-config build-essential libffi-dev
git clone https://github.com/facebookresearch/sam2.git
conda create -n sam2 python>=3.10
conda install -n sam2 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
~/miniconda3/envs/sam2/bin/pip install -e . -t sam2
sh ~/sam2/checkpoints/download_ckpts.sh