import math
import numpy as np
import cv2


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def draw_bodypose_25kp(canvas, candidate, subset):
    # Assuming 25 keypoints, defining their connections (limbSeq) and colors
    stickwidth = 4
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
           [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
           [0, 15], [15, 17], [2, 16], [5, 17], [16, 18], [18, 19], [11, 20],
           [20, 21], [0, 22], [22, 23], [23, 12]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 128, 0], [128, 255, 0], [0, 255, 128],
              [0, 128, 255], [255, 0, 128], [128, 0, 255], [255, 255, 255]]

    # Loop through existing 18 keypoints and draw circles on the canvas
    for i in range(18):
        index = int(subset[0][i])
        if index == -1:
            continue
        x, y = candidate[index][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

        # Additional code for labeling keypoints (optional)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(x), int(y))
        fontScale = 0.8
        fontColor = (0, 0, 0)
        thickness = 2
        lineType = 2

        cv2.putText(canvas, str(i),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

    num_keypoints = len(candidate)
    # print(f"Number of keypoints: {num_keypoints}")
    # Loop through 24 connections to draw limbs
    for i in range(24):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        index = int(subset[0][i])
        if index == -1:
            continue
        x, y = candidate[index][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(x), int(y))
        fontScale = 0.8
        fontColor = (0, 0, 0)
        thickness = 2
        lineType = 2

        cv2.putText(canvas, str(i),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


def save_25kp_json(candidate, subset):
    map25to18 = {
                    0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 5,
                    6: 6,
                    7: 7,
                    8: "mean_8_11",
                    9: 8,
                    10: 9,
                    11: 10,
                    12: 11,
                    13: 12,
                    14: 13,
                    15: 14,
                    16: 15,
                    17: 16,
                    18: 17,
                    19: None,
                    20: None,
                    21: None,
                    22: None,
                    23: None,
                    24: None
                }
    result = dict()
    result["people"] = list()
    result["people"].append(dict())
    result["people"][0]["pose_keypoints_2d"] = list()

    kp17 = dict()
    for i in range(18):
        index = int(subset[0][i])
        if index == -1:
            x, y = 0.0, 0.0
        else:
            x, y = candidate[index][0:2]
        kp17[i] = (x, y)
    kp25 = dict()
    for i in range(25):
        keypoint_index_kp17 = map25to18[i]
        if type(keypoint_index_kp17) is int:
            keypoint_kp17 = kp17[keypoint_index_kp17]
            kp25[i] = keypoint_kp17
        else:
            xi_kp25, yi_kp25 = 0.0, 0.0
            if keypoint_index_kp17 == "mean_8_11":
                xi_kp25 = (kp17[8][0] + kp17[11][0])/2
                yi_kp25 = (kp17[8][1] + kp17[11][1])/2
            elif keypoint_index_kp17 is None:
                xi_kp25 = 0.0
                yi_kp25 = 0.0
            kp25[i] = (xi_kp25, yi_kp25)
        result["people"][0]["pose_keypoints_2d"].append(kp25[i][0])
        result["people"][0]["pose_keypoints_2d"].append(kp25[i][1])
        result["people"][0]["pose_keypoints_2d"].append(0.9)

    return result
