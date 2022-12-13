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


import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=1e-6, verbose=True)

    model.train()
    min_epoch_loss = 99999999
    RLR_counter = 0
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

        if epoch_loss > min_epoch_loss:
            RLR_counter += 1
            print(f'loss not reduced for {RLR_counter} epochs')
        else:
            RLR_counter = 0
        scheduler.step()
        
        # 에폭당 평균 로스들을 구함
        mean_loss = epoch_loss / num_batches
        mean_loss_cls = epoch_loss_cls/num_batches
        mean_loss_angle = epoch_loss_angle/num_batches
        mean_loss_iou = epoch_loss_iou/num_batches
        
        # 딕셔너리 형태로 변환하여 wandb에 로깅
        if log_wandb:
            log_dict = dict()
            log_dict["learning_rate"] = optimizer.param_groups[0]['lr']
            log_dict["mean_loss"] = mean_loss
            log_dict["mean_loss_cls"] = mean_loss_cls
            log_dict["mean_loss_angle"] = mean_loss_angle
            log_dict["mean_loss_iou"] = mean_loss_iou
            wandb.log(log_dict)
        
        print('Mean loss: {:.4f} | Elapsed time: {} | lr: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start), optimizer.param_groups[0]['lr']))
        print(f'Mean cls loss: {mean_loss_cls:.4f} | Mean angle loss: {mean_loss_angle:.4f} | Mean iou loss: {mean_loss_iou:.4f}')

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        min_epoch_loss = min(min_epoch_loss, epoch_loss)

def main(args):
    # 시드 고정
    seed_everything(42)
    
    if args.log_wandb:
        # 본인의 프로젝트를 argparser로 넣으세요, entity는 기존에 사용하던 팀 엔티티를 사용합니다, 실험 이름은 argpaser로 넣으세요.
        wandb.init(project=args.project_name, entity="boostcamp_aitech4_jdp", name=args.exp_name, save_code=True)
    
        # 기본적으로 args로 설정한 모든 값들을 config로 저장합니다.
        wandb.config.update(args)
    
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
