import os
import sys
import math

import numpy as np
import cv2

import argparse

from openvino.inference_engine import IECore

def _exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [_exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([_exp(item) for item in v], v.dtype)
    
    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base
    
    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def IOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
 
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def NMS(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj[1], reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and IOU(obj[0], objs[j][0]) > iou:
                flags[j] = 1
    return keep


def maxPooling2d(input, kernel, stride, padding):
    _, C, H, W = input.shape
    oh = (H - kernel) // stride + 1 + 2*padding
    ow = (W - kernel) // stride + 1 + 2*padding
    output = np.zeros((1, C, oh, ow), dtype=np.float)
    for c in range(C):
        for y in range(padding, oh-padding, stride):
            for x in range(padding, ow-padding, stride):
                m = input[0, c, y:y+kernel, x:x+kernel].max()
                output[0, c, y, x] = m
    return output


def detect(hm, box, landmark, threshold=0.4, nms_iou=0.5):
    hm_pool = maxPooling2d(hm, 3, 1, 1)                # 1,1,240,320
    interest_points = ((hm==hm_pool) * hm)             # screen out low-conf pixels
    flat            = interest_points.ravel()          # flatten
    indices         = np.argsort(flat)[::-1]           # index sort
    scores          = np.array([ flat[idx] for idx in indices ])

    hm_height, hm_width = hm.shape[2:]
    ys = indices // hm_width
    xs = indices %  hm_width
    box      = box.reshape(box.shape[1:])           # 4,240,320
    landmark = landmark.reshape(landmark.shape[1:]) # 10,240,,320

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold: 
            break
        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (_exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append([xyrb, score, box_landmark])
    return NMS(objs, iou=nms_iou)


def drawBBox(image, bbox, color=(0,255,0), thickness=2, textcolor=(0, 0, 0), landmarkcolor=(0, 0, 255)):

    text = f"{bbox[1]:.2f}"
    xyrb = bbox[0]
    x, y, r, b = int(xyrb[0]), int(xyrb[1]), int(xyrb[2]), int(xyrb[3])
    w = r - x + 1
    h = b - y + 1

    cv2.rectangle(image, (x, y, r-x+1, b-y+1), color, thickness, 16)

    border = int(thickness / 2)
    pos = (x + 3, y - 5)
    cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
    cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

    landmark = bbox[2]
    if len(landmark)>0:
        for i in range(len(landmark)):
            x, y = landmark[i][:2]
            cv2.circle(image, (int(x), int(y)), 3, landmarkcolor, -1, 16)


def main(args):

    ie = IECore()
    
    base,ext = os.path.splitext(args.model)
    if ext != '.xml':
        print('Not .xml file is specified ', args.model)
        sys.exit(-1)
    net = ie.read_network(base+'.xml', base+'.bin')
    exenet = ie.load_network(net, 'CPU')

    inblobs =  (list(net.inputs.keys()))
    outblobs = (list(net.outputs.keys()))
    print(inblobs, outblobs)
    # ['x'] ['Conv_525', 'Exp_527', 'Sigmoid_526']

    inshapes  = [ net.inputs [i].shape for i in inblobs  ]
    outshapes = [ net.outputs[i].shape for i in outblobs ]
    print(inshapes, outshapes)
    # 4vga : [[1, 3, 960, 1280]] [[1, 10, 240, 320], [1, 4, 240, 320], [1, 1, 240, 320]]

    # Assign output node idex by checking the number of channels
    for i,outblob in enumerate(outblobs):
        C = outshapes[i][1]
        if C==1:
            hm_idx=i
        if C==4:
            box_idx=i
        if C==10:
            lm_idx=i

    image = cv2.imread(args.input)
    image = cv2.resize(image, (inshapes[0][3], inshapes[0][2]))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1))

    res = exenet.infer({inblobs[0]:img})
    lm  = res[outblobs[lm_idx ]]     # 1,10,h,w
    box = res[outblobs[box_idx]]     # 1,4,h,w
    hm  = res[outblobs[hm_idx ]]     # 1,1,h,w

    objs = detect(hm=hm, box=box, landmark=lm, threshold=0.2, nms_iou=0.5)

    for obj in objs:
        drawBBox(image, obj)

    cv2.imshow('output', image)
    cv2.waitKey(0)

    cv2.imwrite('output.jpg', image)
    print('"output.jpg" is generated')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='selfie.jpg', help='input image file name')
    parser.add_argument('-m', '--model', type=str, default='./dbface.xml', help='FBFace IR model file name (*.xml)')
    args = parser.parse_args()

    main(args)
