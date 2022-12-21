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
from dataset_for_valid import SceneTextDataset2
import validation

from model import EAST
import wandb

# 영동이가 추가한 부분, 이거 없으면 ai_hub dset으로 train시 오류 발생
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
import numpy as np

# default wandb log 기능
# 2022-12-12 17:48 validation, wandb 선택하는 기능 추가함
# 2022-12-13 07:10 ImageFile.LOAD_TRUNCATED_IMAGES 코드라인 추가(버그 발생)
#                  saveinterval default 1로 변경
#                  model 가중치 저장을 백업용으로 이중으로 수행, (저장이름은 실험이름)
# 2022-12-13 08:24 Validation loss 로깅 버그 수정



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
    parser.add_argument('--save_interval', type=int, default=1)
    # wandb 프로젝트와 실험이름을 저장하기 위해 argpaser 추가함
    parser.add_argument('--project_name', type=str, default="trash_project")
    parser.add_argument('--exp_name', type=str, default="jdp")
    # 선택적으로 log를 남기거나 validation할 수 있도록 argparser 추가함
    parser.add_argument('--log_wandb', type=str, default="False")
    parser.add_argument('--log_val', type=str, default="False")
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, log_wandb, project_name, exp_name, log_val):
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
        if log_wandb == "True":
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

            ckpt_fpath_for_backup = osp.join(model_dir, f'{exp_name}.pth')
            torch.save(model.state_dict(), ckpt_fpath_for_backup)


def do_training_with_valid(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                           learning_rate, max_epoch, save_interval, log_wandb, project_name, exp_name, log_val):
    EARLYSTOP_F1_THRESHOLD = 0

    train_dataset = SceneTextDataset2(data_dir, split='train', image_size=image_size, crop_size=input_size)
    valid_dataset = SceneTextDataset2(data_dir, split='valid', image_size=image_size, crop_size=input_size)
    
    train_dataset = EASTDataset(train_dataset)
    valid_dataset = EASTDataset(valid_dataset)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    valid_batch_size = batch_size//4 # 12, 3, 9
    num_batches_valid = math.ceil(len(valid_dataset) / valid_batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    wandb.watch(model, model.criterion, log="all", log_freq=1)

    for epoch in range(max_epoch):
        model.train()
        print('TRAIN...')
        epoch_loss, epoch_start = 0, time.time()
        epoch_loss_cls, epoch_loss_angle, epoch_loss_iou = 0, 0, 0
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train = loss.item()
                
                epoch_loss += loss_train
                epoch_loss_cls += extra_info['cls_loss']
                epoch_loss_angle += extra_info['angle_loss']
                epoch_loss_iou += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        # 에폭당 평균 로스들을 구함
        mean_loss = epoch_loss / num_batches
        mean_loss_cls = epoch_loss_cls/num_batches
        mean_loss_angle = epoch_loss_angle/num_batches
        mean_loss_iou = epoch_loss_iou/num_batches
        
        

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        print(f'Mean cls loss: {mean_loss_cls:.4f} | Mean angle loss: {mean_loss_angle:.4f} | Mean iou loss: {mean_loss_iou:.4f}')
        
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            ckpt_fpath_for_backup = osp.join(model_dir, f'{exp_name}.pth')
            torch.save(model.state_dict(), ckpt_fpath_for_backup)
        
        # train iter 과정이 마무리되면 valid iter 과정을 수행
        model.eval()
        print('VALIDATION...')
        val_epoch_loss = 0
        val_epoch_loss_cls, val_epoch_loss_angle, val_epoch_loss_iou = 0, 0, 0
        with tqdm(total=num_batches_valid) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                loss_val = loss.item()
                val_epoch_loss += loss_val
                val_epoch_loss_cls += extra_info['cls_loss']
                val_epoch_loss_angle += extra_info['angle_loss']
                val_epoch_loss_iou += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        # validset에폭 당 평균 로스들을 구함
        val_mean_loss = val_epoch_loss / num_batches_valid
        val_mean_loss_cls = val_epoch_loss_cls/num_batches_valid
        val_mean_loss_angle = val_epoch_loss_angle/num_batches_valid
        val_mean_loss_iou = val_epoch_loss_iou/num_batches_valid

        

        print(f'Mean cls loss: {val_mean_loss_cls:.4f} | Mean angle loss: {val_mean_loss_angle:.4f} | Mean iou loss: {val_mean_loss_iou:.4f}')

        scheduler.step()

        valid_annot_path = osp.join(data_dir, 'ufo', 'splited_valid.json')

        ## validation 결과를 inference해서 json파일을 만듭니다.
        val_result = validation.do_validation(model, None, valid_annot_path, input_size, 128, None, exp_name)
        f1 = val_result["total"]["hmean"]
        recall = val_result["total"]["recall"]
        precision = val_result["total"]["precision"]

        print(f"[{epoch} EPOCH VALID RESULT] f1: {f1:.2f}, recall: {recall:.2f}, precision: {precision:.2f}")

        # f1 score가 EARLYSTOP_F1_THRESHOLD 넘으면, EARLYSTOP_F1_THRESHOLD를 현재값으로 업데이트하고 가중치를 저장합니다.
        if f1 > EARLYSTOP_F1_THRESHOLD:
            print(f"score update😉! {EARLYSTOP_F1_THRESHOLD} → {f1:.4f}")
            ckpt_fpath_for_best = osp.join(model_dir, f'_bestscore_{exp_name}.pth')
            torch.save(model.state_dict(), ckpt_fpath_for_best)
            EARLYSTOP_F1_THRESHOLD = f1


        # 딕셔너리 형태로 변환하여 wandb에 로깅
        if log_wandb == "True":
            log_dict = dict()
            log_dict["mean_loss"] = mean_loss
            log_dict["mean_loss_cls"] = mean_loss_cls
            log_dict["mean_loss_angle"] = mean_loss_angle
            log_dict["mean_loss_iou"] = mean_loss_iou

            log_dict["valid_mean_loss"] = val_mean_loss
            log_dict["valid_mean_loss_cls"] = val_mean_loss_cls
            log_dict["valid_mean_loss_angle"] = val_mean_loss_angle
            log_dict["valid_mean_loss_iou"] = val_mean_loss_iou

            log_dict["f1"] = f1
            log_dict["recall"] = recall
            log_dict["precision"] = precision

            wandb.log(log_dict)




def main(args):
    # 시드 고정
    seed_everything(42)
    
    if args.log_wandb == "True":
        print("wandb에 로깅합니다")
        # 본인의 프로젝트를 argparser로 넣으세요, entity는 기존에 사용하던 팀 엔티티를 사용합니다, 실험 이름은 argpaser로 넣으세요.
        # 기본적으로 args로 설정한 모든 값들을 config로 저장합니다.
        wandb.init(project=args.project_name, entity="boostcamp_aitech4_jdp", name=args.exp_name)
        wandb.config.update(args)
    else:
        print("wandb에 로깅하지 않습니다")
    
    # ufo dir에 "splited_train.json"과 "splited_valid.json"이 필요하며 이름은 고정되어 있음
    # 이들은 train.json 파일을 이용하여 생성되며 만들기 위해서는
    # utils>split_train_and_valid.ipynb로 생성 가능함.
    if args.log_val == "True":
        print("validation을 수행합니다")
        do_training_with_valid(**args.__dict__)
    else:
        print("validation을 수행하지 않습니다")
        do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
