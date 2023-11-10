# Using Different Pre-Processing methods

## Pose Estimation
Human Pose estimation is required by tryon diffusion to get jg(garment pose keypoints) and jp(human pose keypoints).
We have chosen to use a [Pytorch Implementation](https://github.com/Hzzone/pytorch-openpose) of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to get pose keypoints.

* Pose Estimation code can be found inside `openpose_pytorch`.
* Download [model weights](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG?usp=sharing) as `openpose_pytorch/body_pose_model.pth`.
* Give image path to variable `test_image` in `openpose_pytorch/body_pose.py`.
* Run `openpose_pytorch/body_pose.py`.


## U2Net Human Parsing


## Segformer Human Parsing
