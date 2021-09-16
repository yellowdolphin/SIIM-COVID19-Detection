import argparse
from multiprocessing import cpu_count
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

from models import SiimCovidAuxModel
from dataset import RSNAPneuAuxDataset, classes, rsnapneumonia_classes, chexpert_classes, chest14_classes

from utils import seed_everything

import warnings
warnings.filterwarnings("ignore")

# Don't set default (=>None) if kwarg is defined in cfg!
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/seresnet152d_512_unet.yaml', type=str)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--bs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--patience", default=8, type=int)
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--aux_weight", type=float)
parser.add_argument("--encoder_act", type=str)
parser.add_argument("--dropout_ps", default=[0.20, 0.05], nargs="+", type=float)
parser.add_argument("--restart", type=str, choices='chexpert chest14 rsna siim'.split())

args = parser.parse_args()
print(args)

SEED = args.seed
seed_everything(SEED)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    ckpt_dir = 'rsnapneu_pretrain'
    os.makedirs(ckpt_dir, exist_ok = True)
    print('Train on RSNA-Pneumonia')
    train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/rsnapneumonia_train.csv')
    valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/rsnapneumonia_valid.csv')
    
    if args.frac != 1:
        print(f'Quick training, frac={args.frac}')
        train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
        valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)

    train_dataset = RSNAPneuAuxDataset(
        df=train_df,
        images_dir='.',
        image_size=cfg['aux_image_size'], mode='train')
    valid_dataset = RSNAPneuAuxDataset(
        df=valid_df,
        images_dir='.',
        image_size=cfg['aux_image_size'], mode='valid')

    batch_size = args.bs or cfg['aux_batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset), 
                              num_workers=cpu_count(), drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset), 
                              num_workers=cpu_count())

    print(f'TRAIN: {len(train_loader):5} batches of {batch_size}     = {len(train_loader) * batch_size:6} / {len(train_loader.dataset):6} images')
    print(f'VALID: {len(valid_loader):5} batches of {batch_size} (or less)   => {len(valid_loader.dataset):6} images')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_act_layer = args.encoder_act or cfg['encoder_act_layer'] if 'encoder_act_layer' in cfg else None
    dropout_ps = args.dropout_ps

    encoder_weights = cfg['encoder_weights'] if args.restart is None else None
    if args.restart is None or args.restart.lower() == 'none':
        encoder_pretrained_path = model_pretrained_path = None
        encoder_pretrained_num_classes = model_pretrained_num_classes = None
    elif args.restart.lower() == 'chexpert':
        encoder_pretrained_path = f"chexpert_chest14_pretrain/{cfg['encoder_name']}_{cfg['chexpert_image_size']}_pretrain_step0.pth"
        encoder_pretrained_num_classes = len(chexpert_classes)
        model_pretrained_path = model_pretrained_num_classes = None
    elif args.restart.lower() == 'chest14':
        encoder_pretrained_path = f"chexpert_chest14_pretrain/{cfg['encoder_name']}_{cfg['chest14_image_size']}_pretrain_step1.pth"
        encoder_pretrained_num_classes = len(chest14_classes)
        model_pretrained_path = model_pretrained_num_classes = None
    elif args.restart.lower() == 'rsna':
        encoder_pretrained_path = encoder_pretrained_num_classes = None
        model_pretrained_path = f"rsnapneu_pretrain/{cfg['encoder_name']}_{cfg['aux_image_size']}_{cfg['decoder']}_rsnapneu.pth"
        model_pretrained_num_classes = len(rsnapneumonia_classes)
    elif args.restart.lower() == 'siim':
        siim_models = glob(f"checkpoints/{cfg['encoder_name']}_{cfg['aux_image_size']}_{cfg['decoder']}_aux_fold*.pth")
        assert len(siim_models) == 1, f'several or no siim checkpoints found: \n{siim_models}'
        encoder_pretrained_path = siim_models[0]
        encoder_pretrained_num_classes = len(classes)
        model_pretrained_path = f"rsnapneu_pretrain/{cfg['encoder_name']}_{cfg['aux_image_size']}_{cfg['decoder']}_rsnapneu.pth"
        model_pretrained_num_classes = len(rsnapneumonia_classes)
        if os.path.exists(model_pretrained_path):
            print("Found rsna checkpoint from previous iteration, will use it for cls_head and decoder.")
        else:
            model_pretrained_path = model_pretrained_num_classes = None
        
    model = SiimCovidAuxModel(
        encoder_name=cfg['encoder_name'],
        encoder_weights=encoder_weights,
        encoder_act_layer=encoder_act_layer,
        decoder=cfg['decoder'],
        classes=len(rsnapneumonia_classes),
        in_features=cfg['in_features'],
        dropout_ps=dropout_ps,
        decoder_channels=cfg['decoder_channels'],
        encoder_pretrained_path=encoder_pretrained_path,
        encoder_pretrained_num_classes=encoder_pretrained_num_classes,
        model_pretrained_path=model_pretrained_path, 
        model_pretrained_num_classes=model_pretrained_num_classes)

    if hasattr(model.encoder, 'act1'):
        print("Encoder activation layer:", model.encoder.act1)

    model.to(device)

    cls_criterion = nn.BCEWithLogitsLoss()
    seg_criterion = DiceLoss()
    aux_weight = cfg['aux_weight'] if args.aux_weight is None else args.aux_weight

    lr = args.lr or cfg['aux_init_lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = args.epochs or cfg['aux_epochs']
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    scaler = torch.cuda.amp.GradScaler()

    LOG = '{}/{}_{}_{}_rsnapneu.log'.format(ckpt_dir, cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'])
    CHECKPOINT = '{}/{}_{}_{}_rsnapneu.pth'.format(ckpt_dir, cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'])

    val_loss_min = np.Inf
    if os.path.isfile(LOG):
        os.remove(LOG)
    log_file = open(LOG, 'a')
    log_file.write('epoch, lr, train_cls, train_iou, train_loss, val_cls, val_iou, val_loss\n')
    log_file.close()

    count = 0
    best_epoch = 0

    iou_func = IoU(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None)

    print(f"Training for {epochs} epochs with bs={batch_size}, initial lr={lr}, aux_weight={aux_weight}, seed={SEED}")
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_cls = []
        train_iou = []

        loop = tqdm(train_loader)
        for images, masks, labels in loop:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if cfg['aux_mixup']:
                ### mixup
                lam = np.random.beta(0.5, 0.5)
                rand_index = torch.randperm(images.size(0))
                images = lam * images + (1 - lam) * images[rand_index, :,:,:]
                labels_a, labels_b = labels, labels[rand_index]
                masks_a, masks_b = masks, masks[rand_index,:,:,:]
            
                with torch.cuda.amp.autocast():
                    seg_outputs, cls_outputs = model(images)
                    cls_loss = lam * cls_criterion(cls_outputs, labels_a) + (1 - lam) * cls_criterion(cls_outputs, labels_b)
                    seg_loss = lam * seg_criterion(seg_outputs, masks_a) + (1 - lam) * seg_criterion(seg_outputs, masks_b)
                    loss = aux_weight * cls_loss + (1 - aux_weight) * seg_loss

                    train_iou.append(iou_func(seg_outputs, masks).item())
                    train_loss.append(loss.item())
            else:
                with torch.cuda.amp.autocast():
                    seg_outputs, cls_outputs = model(images)
                    cls_loss = cls_criterion(cls_outputs, labels)
                    seg_loss = seg_criterion(seg_outputs, masks)
                    loss = aux_weight * cls_loss + (1 - aux_weight) * seg_loss

                    train_cls.append(cls_loss.item())
                    train_iou.append(iou_func(seg_outputs, masks).item())
                    train_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description('Epoch {}/{} | LR: {:.5f}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=np.mean(train_loss), iou=np.mean(train_iou))
        train_loss = np.mean(train_loss)
        train_cls = np.mean(train_cls)
        train_iou = np.mean(train_iou)

        model.eval()

        val_loss = 0
        val_cls = 0
        val_iou = 0
        for images, masks, labels in tqdm(valid_loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(), torch.no_grad():
                seg_outputs, cls_outputs = model(images)
                cls_loss = cls_criterion(cls_outputs, labels)
                seg_loss = seg_criterion(seg_outputs, masks)
                loss = aux_weight * cls_loss + (1 - aux_weight) * seg_loss

                val_cls += cls_loss
                val_iou += iou_func(seg_outputs, masks).item()*images.size(0)
                val_loss += loss.item()*images.size(0)

        val_cls /= len(valid_loader.dataset)
        val_iou /= len(valid_loader.dataset)
        val_loss /= len(valid_loader.dataset)

        print('train iou: {:.5f} | train loss: {:.5f} | val_iou: {:.5f} | val_loss: {:.5f}'.format(train_iou, train_loss, val_iou, val_loss))
        log_file = open(LOG, 'a')
        log_file.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
            epoch + 1, optimizer.param_groups[0]['lr'], train_cls, train_iou, train_loss, val_cls, val_iou, val_loss))
        log_file.close()

        if val_loss < val_loss_min:
            print('Valid loss improved from {:.5f} to {:.5f} saving model to {}'.format(val_loss_min, val_loss, CHECKPOINT))
            val_loss_min = val_loss
            best_epoch = epoch + 1
            count = 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            count += 1
        
        if count > args.patience:
            break

        if ('scheduler' in globals()) and scheduler is not None:
            scheduler.step()
    
    log_file = open(LOG, 'a')
    log_file.write('Best epoch {} | val loss min: {}\n'.format(best_epoch, val_loss_min))
    log_file.close()
    print('Best epoch {} | val loss min: {}'.format(best_epoch, val_loss_min))