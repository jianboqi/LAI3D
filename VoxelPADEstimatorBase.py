# coding: utf-8
import os
from Utils import *
import laspy
import numpy as np
import math


class VoxelPADEstimatorBase:
    def __init__(self, filtered_input_files, file_type, resolution, txt_header=None):
        '''
        构造函数
        :param filtered_input_files:  地面滤波之后得文件
        '''
        self.filtered_input_files = filtered_input_files
        self.file_type = file_type
        self.resolution = resolution
        self.txt_header = txt_header

        self.all_pulses = None

        # bounding box 保存的是min_x, min_y, min_z, width, height, depth, num_w, num_h, num_d, max_x, max_y, max_z
        # 分别是：最小x，最小y，最小z，宽度（m），高度，深度，横向网格数量，纵向网格数量，深度方向网格数量
        self.bb = BoundingBox(self.compute_bound_box_multi_file())
        # record the total LAD of each voxel
        # self.out_LADs = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d), dtype=float)
        self.pad = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d))

    def pad_inversion(self, parallel_computing=False):
        pass

    def save_pad(self, output_pad_file_path):
        if self.pad is not None:
            print(" - Save file to ", output_pad_file_path)
            np.save(output_pad_file_path, self.pad)

    def save_lai2d(self, out_lai2d_file_path):
        if self.pad is not None:
            filename, file_extension = os.path.splitext(out_lai2d_file_path)
            if file_extension != ".txt":
                out_lai2d_file_path = out_lai2d_file_path + ".txt"
            f = open(out_lai2d_file_path, "w")
            rows, cols, depths = self.pad.shape
            for row in range(rows):
                for col in range(cols):
                    lads = self.pad[row][col][:]
                    leaf_area = (lads*self.resolution*self.resolution*self.resolution).sum()
                    lai = leaf_area/(self.resolution*self.resolution)
                    f.write("{:7.2f}".format(lai))
                f.write("\n")
            f.close()


    # 计算整个点云的包围盒
    def compute_bound_box_multi_file(self):
        if self.file_type == "las":
            return self.compute_bounding_box_multi_las()
        else:  # text
            total_min = [float('inf'), float('inf'), float('inf')]
            total_max = [-float('inf'), -float('inf'), -float('inf')]
            for file_path in self.filtered_input_files:
                f = open(file_path, 'r')
                for line in f:
                    arr = line.strip().split(" ")
                    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
                    if x < total_min[0]:
                        total_min[0] = x
                    if y < total_min[1]:
                        total_min[1] = y
                    if z < total_min[2]:
                        total_min[2] = z
                    if x > total_max[0]:
                        total_max[0] = x
                    if y > total_max[1]:
                        total_max[1] = y
                    if z > total_max[2]:
                        total_max[2] = z

            # add a small value to be more convenient to do gridding
            width = total_max[0] - total_min[0] + 0.00001
            height = total_max[1] - total_min[1] + 0.00001
            depth = total_max[2] - total_min[2] + 0.00001
            min_x, min_y, min_z = total_min[0], total_min[1], total_min[2]
            num_w = int(math.ceil(width / float(self.resolution)))
            num_h = int(math.ceil(height / float(self.resolution)))
            num_d = int(math.ceil(depth / float(self.resolution)))
            return min_x, min_y, min_z, width, height, depth, num_w, num_h, num_d, total_max[0], total_max[1], \
                   total_max[2]

    def compute_bounding_box_multi_las(self):
        total_min = [float('inf'), float('inf'), float('inf')]
        total_max = [-float('inf'), -float('inf'), -float('inf')]
        for las_file_path in self.filtered_input_files:
            input_File = laspy.read(las_file_path)
            mins = input_File.header.min
            maxs = input_File.header.max
            for i in range(0, 3):
                total_min[i] = min(total_min[i], mins[i])
            for i in range(0, 3):
                total_max[i] = max(total_max[i], maxs[i])
        # add a small value to be more convenient to do gridding
        width = total_max[0] - total_min[0] + 0.00001
        height = total_max[1] - total_min[1] + 0.00001
        depth = total_max[2] - total_min[2] + 0.00001
        min_x, min_y, min_z = total_min[0], total_min[1], total_min[2]
        num_w = int(math.ceil(width / float(self.resolution)))
        num_h = int(math.ceil(height / float(self.resolution)))
        num_d = int(math.ceil(depth / float(self.resolution)))
        return min_x, min_y, min_z, width, height, depth, num_w, num_h, num_d, total_max[0], total_max[1], \
               total_max[2]