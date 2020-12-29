from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os
from natsort import natsorted, ns

import time

def pred(model, img):
    global result_frame
    score_thr = 0.6

    t1 = time.time()
    result = inference_detector(model, img)
    t2 = time.time()

    print("Inference FPS : {}".format(1/(t2-t1)))

    model.show_result(img, result, score_thr=score_thr, wait_time=1, show=True, win_name='inference')

    return result

def read_directory(root_dir, model) :

    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)

    for (sub_root, dirs, files) in os.walk(root_dir):
        dirs.sort()
        files = natsorted(files , alg=ns.IGNORECASE)

        print("## Directory :", sub_root)

        for idx, file_name in enumerate(files):
            result = pred(model=model, img = '{}/{}'.format(sub_root,file_name))

def save_img_result(file_dir, bbox, score_thr, bbox_color):
    model.show_result(file_dir, bbox, score_thr=score_thr, bbox_color=bbox_color, wait_time=1, show=True, win_name='result_frame')

def get_bbox_cord(bbox):
    x1 = int(bbox[0][0][0])
    y1 = int(bbox[0][0][1])
    x2 = int(bbox[0][0][2])
    y2 = int(bbox[0][0][3])

    return np.array([[x1, y1, x2, y2]])


def ready_model(config_file, checkpoint_file):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    return model

def intersection_over_union(box1, box2):
    if len(box2[0][0]) != 5:
        return 0

    x1 = max(box1[0][0][0], box2[0][0][0])
    y1 = max(box1[0][0][1], box2[0][0][1])
    x2 = min(box1[0][0][2], box2[0][0][2])
    y2 = min(box1[0][0][3], box2[0][0][3])

    area_intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (box1[0][0][2] - box1[0][0][0]) * (box1[0][0][3] - box1[0][0][1])
    area_box2 = (box2[0][0][2] - box2[0][0][0]) * (box2[0][0][3] - box2[0][0][1])
    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union
    return iou

def get_best_bbox(result, score_thr):

    bbox = np.array(result)

    if bbox.size == 0:
        return None

    bbox = [bbox[:,np.argmax(bbox[:, :, -1])]]

    if bbox[0][0][-1] > score_thr:
        return bbox
    else:
        return None


if __name__ == "__main__":

    config_file = 'configs/nas_fpn/my_retinanet_r50_nasfpn_crop640_50e_coco.py'
    checkpoint_file = 'checkpoints/epoch_8.pth'
    model = ready_model(config_file, checkpoint_file)

    root_dir = "./data/test"

    start = time.time()
    read_directory(root_dir, model)
    end = time.time()

    print("\n\tTOTAL TIME : {}".format(end-start))
