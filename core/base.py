import open3d as o3d
import numpy as np
import copy


class PointCloud:
    def __init__(self, file_pass):
        self.pcd = o3d.io.read_point_cloud(file_pass)
        self.pcd_full_points = self.pcd.voxel_down_sample(0.05)

    def down_sample(self, voxel_size):
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def estimate_normal(self, radius, max_nn, re_estimate=False):
        if self.pcd.has_normals() and (re_estimate is False):
            print(":: Already have normal")
        else:
            self.pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    def calculate_fpfh(self, radius, max_nn):
        self.pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    def change_all_color(self, color='blue', which_pcd=2):

        COLORS = {'blue': [0, 0.651, 0.929],
                  'yellow': [1, 0.706, 0],
                  'pink': [0.91, 0.65, 0.82],
                  'gray': [0.68, 0.68, 0.68]}

        if which_pcd == 0:
            self.pcd.paint_uniform_color(COLORS[color])
        elif which_pcd == 1:
            self.pcd_full_points.paint_uniform_color(COLORS[color])
        elif which_pcd == 2:
            self.pcd.paint_uniform_color(COLORS[color])
            self.pcd_full_points.paint_uniform_color(COLORS[color])

    def reset_down_sample(self):
        self.pcd = copy.deepcopy(self.pcd_full_points)

    def invert_normal(self):
        """
        invert normal vector
        """
        self.pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(self.pcd.normals) * (-1))

    def change_selected_points_color(self, index, color='pink'):

        COLORS = {'blue': [0, 0.651, 0.929],
                  'yellow': [1, 0.706, 0],
                  'pink': [0.91, 0.65, 0.82],
                  'gray': [0.68, 0.68, 0.68]}

        for idx in index:
            self.pcd.colors[idx] = COLORS[color]

    def transform(self, trans_matrix):
        self.pcd.transform(trans_matrix)
        self.pcd_full_points.transform(trans_matrix)
