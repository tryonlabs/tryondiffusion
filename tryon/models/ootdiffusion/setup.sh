ENV_NAME="ootdiffusion"
PROJECT_DIR="/home/ubuntu/ootdiffusion"

if [ ! -d ~/miniconda3/envs/$ENV_NAME ]; then
  echo "creating conda environment"
  conda create -y -n $ENV_NAME python==3.10
fi

# clone repository
if [ ! -d $PROJECT_DIR ]; then
  echo "cloning OOTDiffusion repository"
  git clone https://github.com/tryonlabs/OOTDiffusion.git $PROJECT_DIR
fi

~/miniconda3/envs/$ENV_NAME/bin/pip install -r $PROJECT_DIR/requirements.txt

if [ ! -d $PROJECT_DIR/checkpoints/ootd ]; then
  echo "downloading checkpoints"

  # download checkpoints
  git clone https://huggingface.co/levihsu/OOTDiffusion ~/ootd-checkpoints
  git clone https://huggingface.co/openai/clip-vit-large-patch14 ~/clip-vit-large-patch14

  mv ~/ootd-checkpoints/checkpoints/* $PROJECT_DIR/checkpoints/
  rm -rf ~/ootd-checkpoints

  mv ~/clip-vit-large-patch14 $PROJECT_DIR/checkpoints/
  rm  -rf ~/clip-vit-large-patch14

fi