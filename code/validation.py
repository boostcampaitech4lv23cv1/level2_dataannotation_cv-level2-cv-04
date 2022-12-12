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


def do_validation(model, ckpt_fpath, valid_annot_path, input_size, batch_size, split='public'):
    print('Inference in progress')
    model.eval()

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
    
    with open("validset_infer_result.json", 'w') as f:
        json.dump(ufo_result, f, indent=4)

    # 현재 문제가 되는 부분
    resDict = deteval_v2.calc_deteval_metrics(ufo_result["images"], valid_anno["images"])
    

    return None

