# Using Different Pre-Processing methods

## Pose Estimation
Human Pose estimation is required by tryon diffusion to get jg(garment pose keypoints) and jp(human pose keypoints).
We have chosen to use a [Pytorch Implementation](https://github.com/Hzzone/pytorch-openpose) of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to get pose keypoints.

* Pose Estimation code can be found inside `openpose_pytorch`.
* Download [model weights](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG?usp=sharing) as `openpose_pytorch/body_pose_model.pth`.
* Give image path to variable `test_image` in `openpose_pytorch/body_pose.py`.
* Run `openpose_pytorch/body_pose.py`.


## U2Net Human Parsing


## U2Net Cloth Segmentation
Run the following command to segment clothes using the U2Net model
```
python segment_cloth_u2net.py --inputs_dir inputs --outputs_dir outputs --checkpoint_path pre_processing/u2net_cloth_seg/checkpoints/cloth_segm.pth 
```
Note: Download the checkpoint file by following the instructions given in the repository https://github.com/wildoctopus/huggingface-cloth-segmentation


## Segformer Human Parsing
