import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from .u2net import RescaleT, ToTensorLab, SalObjDataset, normPRED, load_human_segm_model


def pred_to_image(predictions, image_path):
    im = Image.fromarray(predictions.squeeze().cpu().data.numpy() * 255).convert('RGB')
    image = io.imread(image_path)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    return imo


def segment_human(image_path, output_dir):
    """
    Segment human using U-2-Net
    :param image_path: image path
    :param output_dir: output directory
    """
    model_name = "u2net"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = [image_path]

    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=images,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    net = load_human_segm_model(device, model_name)

    # 2. inference
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", images[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        mask = pred_to_image(pred, images[i_test])
        mask_cv2 = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)

        subimage = cv2.subtract(mask_cv2, cv2.imread(images[i_test]))
        original = Image.open(images[i_test])
        subimage = Image.fromarray(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))

        subimage = subimage.convert("RGBA")
        original = original.convert("RGBA")

        subdata = subimage.getdata()
        ogdata = original.getdata()

        newdata = []
        for i in range(subdata.size[0] * subdata.size[1]):
            if subdata[i][0] == 0 and subdata[i][1] == 0 and subdata[i][2] == 0:
                newdata.append((231, 231, 231, 231))
            else:
                newdata.append(ogdata[i])
        subimage.putdata(newdata)

        subimage.save(os.path.join(output_dir, f"{images[i_test].split(os.sep)[-1].split('.')[0]}.png"))

        del d1, d2, d3, d4, d5, d6, d7
