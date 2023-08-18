from part_assembly.stage3_data import build_sample_broken_train_dataset, build_sample_broken_val_dataset
import jhutil
from tqdm import tqdm

cfg = jhutil.load_yaml("yamls/data_config.yaml")
dataset = build_sample_broken_train_dataset(cfg.data_broken_noisy)
for data in tqdm(dataset):
    continue
dataset = build_sample_broken_val_dataset(cfg.data_broken_noisy)
for data in tqdm(dataset):
    continue