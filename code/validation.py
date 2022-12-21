import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob
import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm
from detect import detect
import deteval_v2

def format_change(infer_format:dict) -> dict:
    metric_format = dict()
    for img_name in infer_format.keys():
        bboxes = []
        for idx, bbox in infer_format[img_name]["words"].items():
            bboxes.append(bbox["points"])
        metric_format[f"{img_name}"] = bboxes
    return metric_format

def do_validation(model, ckpt_fpath, valid_annot_path, input_size, batch_size, split='public', json_file_name='temp'):
    print('Inference in progress')
    model.eval()

    gt_annot_path = valid_annot_path

    image_fnames, by_sample_bboxes = [], []
    with open(valid_annot_path, 'r') as f:
        valid_anno = json.load(f)
    
    img_dir = valid_annot_path.replace("ufo/splited_valid.json", "images")

    # 이미지에 대한 full_paths를 만들고
    img_fpaths = [osp.join(img_dir, i) for i in valid_anno["images"].keys()]

    images = []

    # loop돌면서 inference 한다.
    # 이미지 사이즈는 detect함수에서 변환된다.
    for image_fpath in tqdm(img_fpaths):
        image_fnames.append(osp.basename(image_fpath))
        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []
    
    ## 이미지가 배치 개수로 나눠떨어지지 않는 경우 남은 잔량을 inference 한다.
    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    # 결과를 딕셔너리 형태로 변환한다.
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)
    
    # validation inference 결과를 저장할 폴더를 만듭니다
    os.makedirs("/opt/ml/code/validations", exist_ok=True)
    with open(f"./validations/{json_file_name}.json", 'w') as f:
        json.dump(ufo_result, f, indent=4)

    pred_annot_path = f"./validations/{json_file_name}.json"
    gt_annot_path

    # .json들을 다시 불러옵니다
    with open(pred_annot_path, 'r') as f:
        preds = json.load(f)
    with open(gt_annot_path, 'r') as f:
        gts = json.load(f)
    
    # format을 변경합니다
    preds_for_metric = format_change(preds['images'])
    gts_for_metrics = format_change(gts['images'])

    # metric 계산하고 리턴합니다
    resDict = deteval_v2.calc_deteval_metrics(preds_for_metric, gts_for_metrics)
    return resDict

