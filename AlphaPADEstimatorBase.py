# coding: utf-8
from Utils import *
import math
import laspy


class AlphaPADEstimatorBase:
    def __init__(self, filtered_input_files, file_type, txt_header=None):
        self.filtered_input_files = filtered_input_files
        self.file_type = file_type
        self.txt_header = txt_header

        self.bb = BoundingBox(self.compute_bound_box_multi_file())
        self.pad = None

    def pad_inversion(self, parallel_computing=False):
        pass

    def save_pad(self, output_pad_file_path):
        pass

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

            width = total_max[0] - total_min[0]
            height = total_max[1] - total_min[1]
            depth = total_max[2] - total_min[2]
            min_x, min_y, min_z = total_min[0], total_min[1], total_min[2]
            num_w = -1  # there is no griding in alphashape
            num_h = -1
            num_d = -1
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
        width = total_max[0] - total_min[0]
        height = total_max[1] - total_min[1]
        depth = total_max[2] - total_min[2]
        min_x, min_y, min_z = total_min[0], total_min[1], total_min[2]
        num_w = -1
        num_h = -1
        num_d = -1
        return min_x, min_y, min_z, width, height, depth, num_w, num_h, num_d, total_max[0], total_max[1], \
               total_max[2]
