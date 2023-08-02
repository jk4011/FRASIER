from part_assembly.stage4 import broken_surface_segmentation, load_pointnext

from jhutil import load_yaml
from jhutil import matrix_from_quat_trans, show_multiple_objs
import os
from part_assembly.stage4 import FractureSet


def fracture_assembly():
    pointnext = load_pointnext()

    data_cfg = load_yaml("/data/wlsgur4011/part_assembly/yamls/data_example.yaml")
    train_loader, val_loader = build_geometry_dataloader(data_cfg, use_saved=True)

    i = 0
    for data in train_loader:
        n_iter = 10
        n_obj_threshold = 5
        import jhutil; jhutil.jhprint(1111, data)
        # break
        pcd_list = data["pcd"]
        if len(pcd_list) > n_obj_threshold:
            continue

        import jhutil; jhutil.jhprint(0000, i, n_iter)
        if i == n_iter:
            break
        i += 1
        # pcd_list = [segmentation(pcd) for pcd in pcd_list]
        # TODO: point cloud는 gpu memory에서만 다루기
        # pcd_list = to_cuda(pcd_list)

        result = FractureSet(pcd_list).search()
        final_node = result.nodes[0]
        pcd_xored = final_node.pcd

        import jhutil; jhutil.jhprint(1111, final_node.merge_state)
        import jhutil; jhutil.jhprint(2222, final_node.T_dic)
        import jhutil; jhutil.jhprint(5555, result.n_removed)

        transforms_data = []
        for i in range(len(pcd_list)):
            quat = data['quat'][i]
            trans = data['trans'][i]
            matrix = matrix_from_quat_trans(quat, trans).float()
            transforms_data.append(matrix.inverse())

        transforms_restored = [final_node.T_dic[i] @ transforms_data[i] for i in range(len(pcd_list))]

        folder_path = data['dir_name']

        file_list = os.listdir(folder_path)
        file_list = [os.path.join(folder_path, file) for file in file_list]
        show_multiple_objs(file_list, transformations=transforms_restored)


