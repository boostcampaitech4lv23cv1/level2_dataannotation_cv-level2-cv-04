import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import wandb

import random
import numpy as np


def seed_everything(seed:int = 42):
    """재현을 하기 위한 시드 고정 함수
    Args:
        seed (int, optional): 시드. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    # wandb 프로젝트와 실험이름을 저장하기 위해 argpaser 추가함
    parser.add_argument('--log_wandb', type=bool, default=False)
    parser.add_argument('--project_name', type=str, default="trash_project")
    parser.add_argument('--exp_name', type=str, default="jdp")
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, log_wandb, project_name, exp_name):
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        epoch_loss_cls, epoch_loss_angle, epoch_loss_iou = 0, 0, 0
        
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_loss_cls += extra_info['cls_loss']
                epoch_loss_angle += extra_info['angle_loss']
                epoch_loss_iou += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        
        # 에폭당 평균 로스들을 구함
        mean_loss = epoch_loss / num_batches
        mean_loss_cls = epoch_loss_cls/num_batches
        mean_loss_angle = epoch_loss_angle/num_batches
        mean_loss_iou = epoch_loss_iou/num_batches
        
        # 딕셔너리 형태로 변환하여 wandb에 로깅
        if log_wandb:
            log_dict = dict()
            log_dict["mean_loss"] = mean_loss
            log_dict["mean_loss_cls"] = mean_loss_cls
            log_dict["mean_loss_angle"] = mean_loss_angle
            log_dict["mean_loss_iou"] = mean_loss_iou
            wandb.log(log_dict)
        
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        print(f'Mean cls loss: {mean_loss_cls:.4f} | Mean angle loss: {mean_loss_angle:.4f} | Mean iou loss: {mean_loss_iou:.4f}')

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    # 시드 고정
    seed_everything(42)
    
    # 본인의 프로젝트를 argparser로 넣으세요, entity는 기존에 사용하던 팀 엔티티를 사용합니다, 실험 이름은 argpaser로 넣으세요.
    wandb.init(project=args.project_name, entity="boostcamp_aitech4_jdp", name=args.exp_name)
    
    # 기본적으로 args로 설정한 모든 값들을 config로 저장합니다.
    wandb.config.update(args)
    
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
