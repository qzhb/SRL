"""Main training/test program for RULSTM"""
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import socket
from os.path import join
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import os

from args import parse_args
from train import train
from model import get_model
from validation import validation
from fusion_validation import fusion_validation
from dataloaders.EPIC import SequenceDataset
from utils import Logger, topk_accuracy

pd.options.display.float_format = '{:05.2f}'.format


def get_loader(args, mode, override_modality = None):
    if args.modality != 'fusion':
        path_to_lmdb = join(args.path_to_data, args.modality)
        kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'action_samples': args.S_ant if args.task == 'early_recognition' else None,
        'past_features': args.task == 'anticipation',
        'sequence_length': args.S_enc + args.S_ant,
        'label_type': ['verb', 'noun', 'action'],
        }
    
        _set = SequenceDataset(**kargs)

        return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=mode == 'training')
    else:
        Loader = []
        for m in ['rgb', 'flow', 'obj']:
            path_to_lmdb = join(args.path_to_data, m)
            kargs = {
                'path_to_lmdb': path_to_lmdb,
                'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
                'time_step': args.alpha,
                'img_tmpl': args.img_tmpl,
                'action_samples': args.S_ant if args.task == 'early_recognition' else None,
                'past_features': args.task == 'anticipation',
                'sequence_length': args.S_enc + args.S_ant,
                'label_type': ['verb', 'noun', 'action'],
            }
            
            _set = SequenceDataset(**kargs)
            Loader.append(DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, shuffle=mode == 'training'))

        return Loader

def load_checkpoint(args, model, best=False):
    if best:
        chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, args.resume_timestamp, args.exp_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, args.resume_timestamp, args.exp_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf

def main():
    ## set parameters
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'EPIC':
        args.num_class = 2513
        args.verb_num_class = 125
        args.noun_num_class = 352
    
    args.exp_name = f"{args.model_name}-{args.task}_{args.alpha}_{args.S_enc}_{args.S_ant}_{args.modality}"
    
    if args.mode == 'train':
        args.timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        args.save_path = join(args.path_to_results, args.dataset, args.model_name, args.timestamp)
        os.makedirs(args.save_path)

        trainval_logger = Logger(join(args.save_path, 'trainval.log'), ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])

        # save parameters
        with open(os.path.join(args.save_path, 'args.json'), 'w') as args_file:
            json.dump(vars(args), args_file)
        print(args)

        # Logging Tensorboard
        log_dir = os.path.join(args.save_path, 'tensorboard', 'runs', socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir, comment='-params')

    model = get_model(args)
    if type(model) == list:
        model = [m.to(args.device) for m in model]
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        model.to(args.device)

    if args.mode == 'train':
        loaders = {m: get_loader(args, m) for m in ['training', 'validation']}

        if args.resume_timestamp:
            start_epoch, _, start_best_perf = load_checkpoint(args, model)
        else:
            start_epoch = 0
            start_best_perf = 0
   
        train(args, model, loaders, optimizer, args.epochs, start_epoch, start_best_perf, trainval_logger, writer)
        
        writer.close()

    elif args.mode == 'validate':
        if args.modality == 'fusion':
            print('Fuison validation')    
            loader = get_loader(args, 'validation')
            fusion_validation(args, model, loader)
        else: 
            epoch, perf, _ = load_checkpoint(args, model, best=True)
            print(f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")    
            loader = get_loader(args, 'validation')
            validation(args, model, loader)
    

if __name__ == '__main__':
    main()
