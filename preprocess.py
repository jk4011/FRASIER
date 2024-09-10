from part_assembly.stage1_data import build_sample_20k_train_dataset, build_sample_20k_val_dataset, build_sample_20k_dataloader
from part_assembly.stage3_data import build_sample_broken_train_dataset, build_sample_broken_val_dataset, build_sample_broken_dataloader
from part_assembly.test_data import build_sample_dense_dataloader
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from time import time
from tqdm import tqdm


def preprocess_20k_data():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=None)
    parser.add_argument('--min_numpart', type=int, default=None)

    # parse
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_config.yaml")
    if args.overfit is not None:
        cfg.data_20k.overfit = args.overfit
        cfg.data_dense.overfit = args.overfit

    # train_loader, val_loader = build_sample_20k_dataloader(cfg.data_20k)
    train_dataset = build_sample_20k_train_dataset(cfg.data_20k)
    val_dataset = build_sample_20k_val_dataset(cfg.data_20k)

    pbar = tqdm(enumerate(train_dataset), total=len(train_dataset))
    for i, data in tqdm(pbar):
        print(data)
        # jhutil.jhprint(1111, data)
        print("\n\n")
        if i == 3:
            break


def preprocess_dense_data():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=None)
    parser.add_argument('--min_numpart', type=int, default=None)

    # parse
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_config.yaml")
    if args.overfit is not None:
        cfg.data_20k.overfit = args.overfit
        cfg.data_dense.overfit = args.overfit

    val_loader = build_sample_dense_dataloader(cfg.data_dense)

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, data in tqdm(pbar):
        print(data.keys())
        print("\n\n")
        if i == 3:
            break


def preprocess_broken_noisy_data():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=None)
    parser.add_argument('--min_numpart', type=int, default=None)
    
    # parse
    args = parser.parse_args()
    
    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_config.yaml")
    if args.overfit is not None:
        cfg.data_broken_noisy.overfit = args.overfit
    train_dataset = build_sample_broken_train_dataset(cfg.data_broken_noisy)
    val_dataset = build_sample_broken_val_dataset(cfg.data_broken_noisy)

    pbar = tqdm(enumerate(val_dataset), total=len(val_dataset))
    for i, data in tqdm(pbar):
        print(data.keys())
        print("\n\n")
        if i == 3:
            break
    

if __name__ == "__main__":
    # preprocess_20k_data()
    # preprocess_dense_data()
    preprocess_broken_noisy_data()