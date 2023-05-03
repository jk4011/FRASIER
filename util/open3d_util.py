import open3d as o3d
import numpy as np
import copy
import torch

def open3d_preprocess_pcd(pcd_raw, normal=None, voxel_size=0.01):

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_raw)

    radius_normal = voxel_size * 2

    if normal is None:
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    else:
        pcd.normals = o3d.utility.Vector3dVector(normal)
    
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, # pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_down, pcd_fpfh


def open3d_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size, normal_angle=0.05):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    # target_down.normals

    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
            #     0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_angle),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold * 0.1),
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))
    print(1111, result)
    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))

    return result.transformation


def open3d_icp(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    target.normals = o3d.utility.Vector3dVector(-torch.Tensor(target.normals))
    trans_init = np.identity(4) # result_ransac.transformation
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    target.normals = o3d.utility.Vector3dVector(-torch.Tensor(target.normals))
    return result.transformation


def open3d_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))
    return result.transformation






