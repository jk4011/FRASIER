import torch
tmp = torch.randn(1).cuda()

import sys
# sys.path.append("../")
sys.path.append("./src/pointnext")
sys.path.append("./src/geotransformer")

import argparse
import os

from jhutil import load_yaml
# import part_assembly
from part_assembly.stage4 import FractureSet
from part_assembly.test_data import build_sample_dense_dataloader
from part_assembly.stage1_data import build_sample_20k_test_loder
from part_assembly.stage1 import broken_surface_segmentation, load_pointnext
from jhutil import matrix_from_quat_trans
from tqdm import tqdm


def run_full_pipe():

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_process", type=int, default=7)
    parser.add_argument("--process_rank", type=int, default=0)
    args = parser.parse_args()

    # load test loader
    cfg = load_yaml("/data/wlsgur4011/part_assembly/yamls/data_config.yaml")
    test_loader_dense = build_sample_dense_dataloader(cfg.data_dense)

    # load stage1 model
    pointnext = load_pointnext()

    tbar = tqdm(enumerate(test_loader_dense), total=len(test_loader_dense))

    for i, data in tbar:
        
        if i % args.total_process != args.process_rank:
            continue
        dir_path = os.path.join(data['dir_name'][0].replace("raw", "processed"), "transforms.pt")
        
        if os.path.exists(dir_path):
            continue
        
        n = len(data['sample'])
        # TODO: 현재 2분정도 걸리는데 더 최적화 해야 할 듯

        pcd_list = broken_surface_segmentation(pointnext, data, all_broken_threshold=1024)
        import jhutil; jhutil.jhprint(1111, pcd_list, list_one_line=False)
        # pcd_list = [pcd[idx].float() for pcd, idx in zip(data['sample'], data['broken_label'])]

        result = FractureSet(pcd_list).search()
        final_node = result.fracs[0]

        transforms_data = []

        for j in range(n):
            quat = data['quats'][j].squeeze()
            trans = data['trans'][j].squeeze()
            matrix = matrix_from_quat_trans(quat, trans).float()
            transforms_data.append(matrix.inverse())

        transforms_restored = [final_node.T_dic[frozenset([j])] @ transforms_data[j] for j in range(n)]
        transforms_restored = [transforms_restored[0].inverse() @ T for T in transforms_restored]

        folder_path = data['dir_name'][0]

        file_list = os.listdir(folder_path)
        file_list = [os.path.join(folder_path, file) for file in file_list]
        # show_multiple_objs(file_list, transformations=gt_transforms_restored, scale=7)

        dir_path = os.path.join(data['dir_name'][0].replace("raw", "processed"), "transforms.pt")
        torch.save(transforms_restored, dir_path)
        print(f"saved to {dir_path}")


if __name__ == "__main__":
    run_full_pipe()
