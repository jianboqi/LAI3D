# coding: utf-8
import os
from Pulse import *
from Utils import *
from VoxelPADEstimatorBase import VoxelPADEstimatorBase
from collections import Counter


def weighted_num_return(point_arr):
    num_return_counter = Counter(point_arr)
    num = 0
    for key in num_return_counter.keys():
        num += num_return_counter[key] * (1 / key)
    return num


class VoxelPADEstimatorByColumnPoint(VoxelPADEstimatorBase):
    """
    This estimator estimates PAD with discrete point cloud, i.e., does not using intensity information
    """

    def __init__(self, filtered_input_files, file_type, resolution, txt_header=None):
        super().__init__(filtered_input_files, file_type, resolution, txt_header)

    def pad_inversion(self, parallel_computing=False):
        if parallel_computing:
            print(" - INFO: PADEstimatorByPoints does not support parallel computing.")

        print(" - Scene dimensions: X:%.2f Y:%.2f Z:%.2f" % (self.bb.width, self.bb.height, self.bb.depth))
        print(" - Matrix dimensions: ", self.bb.num_w, self.bb.num_h, self.bb.num_d)
        all_points = self.read_merged_points()  # [X Y Z classification num_returns]
        all_ground_points = all_points[all_points[:, 3] == 2]
        all_veg_points = all_points[all_points[:, 3] == 1]
        box_min = np.array([self.bb.min_x, self.bb.min_y, self.bb.min_z])
        ground_indices = np.floor((all_ground_points[:, 0:3] - box_min)/self.resolution)
        veg_indices = np.floor((all_veg_points[:, 0:3] - box_min) / self.resolution)
        total_pai = 0
        for row in range(self.bb.num_h):
            for col in range(self.bb.num_w):
                veg_idx_bool = (veg_indices[:, 1] == (self.bb.num_h-1-row)) & (veg_indices[:, 0] == col)
                column_veg_points = all_veg_points[veg_idx_bool]
                column_veg_z_idx = veg_indices[veg_idx_bool][:, 2]
                num_veg = len(column_veg_points)
                ground_idx_bool = (ground_indices[:, 1] == (self.bb.num_h - 1 - row)) & (ground_indices[:, 0] == col)
                column_ground_points = all_ground_points[ground_idx_bool]
                num_ground = weighted_num_return(column_ground_points[:, 4])
                if num_ground == 0:
                    num_ground = 1
                if num_veg > 0:
                    # raw_num_veg = len(column_veg_points)
                    # transmittance = float(num_ground) / (num_ground+num_veg)
                    # pai_column = -2 * np.log(transmittance)
                    # distribute pai into vertical voxels according to point density
                    v_unique_idx = set(column_veg_z_idx)
                    for v_idx in v_unique_idx:
                        num_veg_voxel = weighted_num_return(column_veg_points[column_veg_z_idx == v_idx][:,4])
                        num_veg_voxel_below = weighted_num_return(column_veg_points[column_veg_z_idx < v_idx][:,4])
                        transmittance = float(num_veg_voxel_below + num_ground) / (num_ground + num_veg_voxel_below +
                                                                                   num_veg_voxel)
                        pai_voxel = -2 * np.log(transmittance)
                        pad_voxel = pai_voxel / self.resolution
                        self.pad[row][col][int(v_idx)] = pad_voxel
                        total_pai += pai_voxel * self.resolution * self.resolution
        print(" - Plot LAI: ", total_pai / float(self.bb.num_w * self.bb.num_h * self.resolution * self.resolution))
        self.save_lai2d("lai2d.txt")

    def read_merged_points(self):
        all_points = None
        for filtered_input_file in self.filtered_input_files:
            if self.file_type == "las":
                in_file = laspy.read(filtered_input_file)
                in_xyz_totals = np.vstack((in_file.x, in_file.y, in_file.z,
                                           in_file.classification, in_file.num_returns)).transpose()
                if all_points is None:
                    all_points = in_xyz_totals
                else:
                    all_points = np.vstack((all_points, in_xyz_totals))
            elif self.file_type == "txt":
                if self.txt_header is None:
                    print("Header is needed for txt format")
                    import sys
                    sys.exit(0)

                in_xyz_totals = []
                headerDic = dict()
                for i in range(0, len(self.txt_header)):
                    headerDic[self.txt_header[i]] = i

                f = open(filtered_input_file, 'r')
                for line in f:
                    arr = line.strip().split(" ")
                    in_xyz_totals.append(
                        [float(arr[headerDic['x']]), float(arr[headerDic['y']]), float(arr[headerDic['z']]),
                         float(arr[headerDic['c']]), float(arr[headerDic['n']])])
                in_xyz_totals = np.array(in_xyz_totals)
                if all_points is None:
                    all_points = in_xyz_totals
                else:
                    all_points = np.vstack((all_points, in_xyz_totals))
        return all_points
