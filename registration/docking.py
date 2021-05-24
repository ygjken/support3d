import numpy as np
import open3d as o3d


class Parameter:
    """This is the structure to set docking() function's parameter
    """

    def __init__(self, voxel_size, fpfh_radius,
                 global_max_corres_distance, global_c_checker_normal,
                 icp_max_corres_distance) -> None:
        self.voxel_size = voxel_size
        # もともとvoxel_size*8
        self.fpfh_radius = fpfh_radius
        # もともとvoxel_size*1.5
        self.corres_max_corres_distance = global_max_corres_distance
        # もともと6
        self.global_c_checker_normal = global_c_checker_normal
        # もともとvoxel_size*0.4
        self.icp_max_corres_distance = icp_max_corres_distance


def one_point_matching(source, target, source_idx):
    matched_points, matched_index = fpfh_matching_top_n(
        source.pcd, target.pcd, source.pcd_fpfh, target.pcd_fpfh, source_idx, 20)

    clustered_points, clustered_index = matched_feature_clustering(
        matched_points)

    target_idxs = np.array(matched_index)[clustered_index]
    corr_indexs = np.array(
        [np.full(len(target_idxs), source_idx), target_idxs])

    return corr_indexs.T, clustered_points


def fpfh_matching_top_n(source, target, source_fpfh, target_fpfh, source_index, topN):
    """this is the function that find top-n points 
    in the target that have FPFH features similar 
    to the points of interest in the source.
    Args:
        source (PointCloud): source point-cloud
        target (PointCloud): target point-cloud
        source_fpfh (Feature): source fpfh-feature
        target_fpfh (Feature): target fpfh-feature
        source_index (list): point's index which will be searched in source
        topN (int): [description]
    Returns:
        points(np.array): point-clouds(x, y, z) which are matched
        idx(list): point-clouds's indexs
    """
    # make KDtree
    tree = o3d.geometry.KDTreeFlann(target_fpfh)
    [_, idx, _] = tree.search_knn_vector_xd(
        source_fpfh.data[:, source_index], topN)

    # points[0] is query
    points = [source.points[source_index]]

    # points[1:] are matched points in target
    for i in idx:
        points.append(target.points[i])
    points = np.array(points)

    return points, list(idx)


def matched_feature_clustering(points):
    """
    Input:
        points: numpy.array
    Return:
        clustered_points: numpy.array
            clustered_points[0]
            clustered_points[1:]
        index: numpy.array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[1:])

    line = list(pcd.cluster_dbscan(eps=2.0, min_points=1))
    index = [line.index(i) for i in set(list(line))]

    selected_points = np.asarray(pcd.points)[index]
    clustered_points = points[0].reshape(1, 3)
    clustered_points = np.append(clustered_points, selected_points, axis=0)

    return clustered_points, index


def docking(source, target, param: Parameter):
    # compute feature
    source.down_sample(param.voxel_size)
    target.down_sample(param.voxel_size)

    # source.change_all_color(color="blue", which_pcd=2)
    # target.change_all_color(color="yellow", which_pcd=2)

    target.estimate_normal(param.voxel_size * 2, 30, True)
    target.invert_normal()
    source.estimate_normal(param.voxel_size * 2, 30, True)

    target.calculate_fpfh(max(param.fpfh_radius, param.voxel_size), 750)
    source.calculate_fpfh(max(param.fpfh_radius, param.voxel_size), 750)

    # top-n fpfh matching
    corr = np.array([], dtype=np.int).reshape(0, 2)

    for source_idx in range(len(source.pcd.points)):
        corr_indexs, _ = one_point_matching(source, target, source_idx)
        corr = np.append(corr, corr_indexs, axis=0)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source.pcd,
        target.pcd,
        o3d.utility.Vector2iVector(corr),
        param.corres_max_corres_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(3.14 / param.global_c_checker_normal),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(param.corres_max_corres_distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000000))

    # global registration done
    # source.transform(result.transformation)

    # local registration
    result = o3d.pipelines.registration.registration_icp(
        source.pcd_full_points,
        target.pcd_full_points,
        param.icp_max_corres_distance,
        result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    source.transform(result.transformation)

    return result
