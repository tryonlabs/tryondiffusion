from torchvision.transforms.v2 import GaussianBlur, PILToTensor


def add_gaussian_blur(img):
    """
    Add Gaussian Blur
    :param img: input image
    :return: noisy image
    """
    gb = GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 0.6))
    img = PILToTensor()(img)
    return gb(img)
