from mmdet.apis import init_detector, inference_detector
# import mmcv
# import torch

import cv2
import pandas as pd
import numpy as np
import os
import copy
from natsort import natsorted, ns

import time

def pred(model, img):
    global result_frame
    score_thr = 0.6

    t1 = time.time()
    result = inference_detector(model, img)
    t2 = time.time()

    print("Inference FPS : {}".format(1/(t2-t1)))

    model.show_result(img, result, score_thr=score_thr, wait_time=0, show=True, win_name='inference')

    return result

def read_directory(root_dir, model) :
    
    score_thr = 0.6

    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)

    for (sub_root, dirs, files) in os.walk(root_dir):
        dirs.sort()
        files = natsorted(files , alg=ns.IGNORECASE)
        
        # if sub_root == root_dir:
        #     continue

        print("## Directory :", sub_root)
        
        recent_iou_list = np.zeros((10,))
        latest_bbox = None
        vip_bbox = None
        
        bbox_list = np.empty((1,4), dtype=np.int16)
        image_name_list = np.empty((0,))
        answer_list = np.empty((0, 5))

        for idx, file_name in enumerate(files):
            print(idx, file_name)
            result = pred(model=model, img = '{}/{}'.format(sub_root,file_name))
            
            if len(result[0]) > 0:
                iou_list = np.empty((0,))
                for pred_bboxes in result:
                    if(latest_bbox is not None):
                        iou_list = np.append(iou_list, intersection_over_union([pred_bboxes], latest_bbox))
                    else:
                        iou_list = np.append(iou_list, 0.90)

                temp_idx = np.argmax(iou_list)
                pred_bbox = [np.array([result[0][temp_idx]])]

                iou = iou_list[temp_idx]
                recent_iou_list[idx % len(recent_iou_list)] = iou

                if recent_iou_list.mean() > 0.90:
                    vip_bbox = copy.deepcopy(latest_bbox)
                    recent_iou_list = np.zeros((12,))

                if iou > 0.90:
                    if vip_bbox is not None:
                        save_img_result(sub_root + '/' + file_name, vip_bbox, score_thr, "red")
                        latest_bbox = copy.deepcopy(vip_bbox)
                    else:
                        save_img_result(sub_root + '/' + file_name, pred_bbox, score_thr, "yellow")
                        latest_bbox = copy.deepcopy(pred_bbox)

                elif iou >= 0.5:
                    save_img_result(sub_root + '/' + file_name, pred_bbox, score_thr, "green")
                    latest_bbox = copy.deepcopy(pred_bbox)

                elif iou >= 0.1:
                    save_img_result(sub_root + '/' + file_name, pred_bbox, score_thr, "cyan")
                    latest_bbox = copy.deepcopy(pred_bbox)

                else:
                    save_img_result(sub_root + '/' + file_name, pred_bbox, score_thr, "cyan")
                    latest_bbox = copy.deepcopy(pred_bbox)
                    vip_bbox = None

            elif len(result[0]) == 0:
                if vip_bbox is not None:
                    save_img_result(sub_root + '/' + file_name, vip_bbox, score_thr, "magenta")
                    latest_bbox = copy.deepcopy(vip_bbox)

                elif latest_bbox is not None:
                    save_img_result(sub_root + '/' + file_name, latest_bbox, score_thr, "white")

            if latest_bbox is not None:
                bbox_list = np.append(bbox_list, get_bbox_cord(latest_bbox), axis=0)
                image_name_list = np.append(image_name_list, file_name)
            else:
                # latest_bbox = copy.deepcopy(pred_bbox)
                bbox_list = np.append(bbox_list, np.array([[300, 150, 500, 300]]), axis=0)
                image_name_list = np.append(image_name_list, file_name)

        set_answer_file(answer_list)
        
def set_answer_file(answer_list):
    answer_csv = pd.DataFrame(
        [answer_data for answer_data in answer_list]
    )
    answer_csv.to_csv('t1_res_0122.csv', index=False, mode='a', header=False)
    print('=*=*=정답기록*=*=')
    
def get_bbox_cord(bbox):
    x1 = int(bbox[0][0][0])
    y1 = int(bbox[0][0][1])
    x2 = int(bbox[0][0][2])
    y2 = int(bbox[0][0][3])

    return np.array([[x1, y1, x2, y2]])

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

#결과물을 저장하는 메소드
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
    
    answer_file_name = './t1_res_0122.csv'
    
    if not os.path.isfile(answer_file_name):
        print("기존 정답파일 존재치 않음")
        answer_csv = pd.DataFrame(
            columns=['file_name', 'left_x', 'left_y', 'right_x', 'right_y']
        )
        answer_csv.to_csv('t1_res_0122.csv', index=False, mode='a')
        print("새 정답파일 생성 : {}\n\n".format(answer_file_name))


    config_file = 'configs/nas_fpn/my_retinanet_r50_nasfpn_crop640_50e_coco.py'
    checkpoint_file = 'checkpoints/epoch_8.pth'
    model = ready_model(config_file, checkpoint_file)

    root_dir = "./data/test"

    start = time.time()
    read_directory(root_dir, model)
    end = time.time()

    print("\n\tTOTAL TIME : {}".format(end-start))
    
    print('=*=*=*=*=*=*=*정 답 추 출 완 료*=*=*=*=*=*=*=')
