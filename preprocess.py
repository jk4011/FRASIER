import sys
sys.path.append("src/multi-part-assembly/")


from multi_part_assembly.datasets.geometry_data import build_geometry_dataset, build_geometry_dataloader
from multi_part_assembly.datasets.geometry_data import GeometryPartDataset
import jhutil
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from time import time
from tqdm import tqdm

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=None)
    parser.add_argument('--min_numpart', type=int, default=None)

    # parse
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    cfg = jhutil.load_yaml("yamls/data_example.yaml")
    if args.overfit is not None:
        cfg.data.overfit = args.overfit
    if args.min_numpart is not None:
        cfg.data.min_numpart = args.min_numpart

    train_set, val_set = build_geometry_dataset(cfg)
    
    for data in tqdm(train_set):
        pass
