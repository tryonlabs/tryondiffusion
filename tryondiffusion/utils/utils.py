import cv2
import math
import numbers
import os

from torch import nn
from torch.nn import functional as F


def load_pose_embed(file_path):
    embed = torch.load(file_path)
    return embed


def read_img(img_path):
    img = cv2.imread(img_path)
    return img


def write_img(img, folder_path, img_name):
    path = os.path.join(folder_path, img_name)
    cv2.imwrite(path, img)


class GaussianSmoothing(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10?u=tanay_agrawal
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, conv_dim):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * conv_dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * conv_dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if conv_dim == 1:
            self.conv = F.conv1d
        elif conv_dim == 2:
            self.conv = F.conv2d
        elif conv_dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(conv_dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name, "images"), exist_ok=True)


if __name__ == "__main__":
    import torch
    import time

    torch.manual_seed(1)
    smoothing = GaussianSmoothing(1, 3, 0.6, 2)
    input = torch.rand(1, 1, 2, 2)
    # print(input)
    input_p = F.pad(input, (1, 1, 1, 1), mode='reflect')
    since = time.time()
    output = smoothing(input_p)
    # print(output)
    print(time.time() - since)
    # smoothing = GaussianSmoothing(1, 3, (0.2, 0.6), 1)
    # input = torch.arange(1, 4)[None, None, :].to(torch.float32)
    # print(input)
    # input = F.pad(input, (1, 1), mode='reflect')
    # output = smoothing(input)
    # print(output)
