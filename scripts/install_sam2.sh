ENV_NAME="sam2"
sudo apt-get -y update && sudo apt-get install -y --no-install-recommends ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev pkg-config build-essential libffi-dev
git clone https://github.com/facebookresearch/sam2.git ~/$ENV_NAME
conda create -n $ENV_NAME python=3.10
conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
~/miniconda3/envs/$ENV_NAME/bin/pip install -y -e ~/$ENV_NAME
sh ~/$ENV_NAME/checkpoints/download_ckpts.sh
mv sam2.1_hiera_base_plus.pt ~/$ENV_NAME/checkpoints/
mv sam2.1_hiera_large.pt ~/$ENV_NAME/checkpoints/
mv sam2.1_hiera_small.pt ~/$ENV_NAME/checkpoints/
mv sam2.1_hiera_tiny.pt ~/$ENV_NAME/checkpoints/
