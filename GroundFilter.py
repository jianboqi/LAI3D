# coding: utf-8
# author: Jianbo QI
# date: 2019-6-26
# 对点云数据进行滤波，根据文件格式分别进行处理，能否同时处理多个文件

import laspy
import CSF
import numpy as np
import os
import sys


class GroundFilter:
    # 当文件类型为las时，txtHeader为None
    def __init__(self, rigidness, input_files, file_ype, txt_header=None):
        self.input_files = input_files  # 输入文件列表
        self.file_ype = file_ype  # 文件类型
        self.rigidness = rigidness  # 布料硬度
        self.txt_header = txt_header  # 文件类型为txt时的文件头

        self.csf = CSF.CSF()  # 构造CSF
        self.csf.params.cloth_resolution = 0.4
        self.csf.params.bSlopeSmooth = True
        self.csf.params.rigidness = self.rigidness
        self.csf.params.class_threshold = 0.3
        self.csf.params.rasterization_mode = 1
        self.csf.params.rasterization_window_size = 10
        self.csf.downsampling_window_num = 5

    def do_filtering(self):
        if self.file_ype == "las":
            self.do_filtering_las()
        if self.file_ype == "txt":
            self.do_filtering_txt()

    def do_filtering_las(self):
        index = 0
        lengths = []
        xyz_total = None
        path_flags = []
        opened_files = []
        for input_file_path in self.input_files:
            input_file = laspy.read(input_file_path)
            opened_files.append(input_file)
            xyz = input_file.xyz
            single_length = xyz.shape[0]
            lengths.append(single_length)
            if index == 0:
                xyz_total = xyz
            else:
                xyz_total = np.vstack((xyz_total, xyz))
            path_flags += [index for i in range(single_length)]
            index += 1

        accumulated_lengths = [0]
        for i in range(0, len(lengths)):
            accumulated_lengths.append(accumulated_lengths[i] + lengths[i])

        self.csf.setPointCloud(xyz_total)
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        self.csf.do_filtering(ground, non_ground)

        # 获取output文件，默认保存到原数据所在路径下的csf_filtered文件夹下
        output_dir = os.path.join(os.path.dirname(self.input_files[0]), "csf_filtered")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_las_files = []
        for input_file_path in self.input_files:
            output_las_file = os.path.join(output_dir, os.path.basename(input_file_path))
            output_las_files.append(output_las_file)

        index = 0
        for i in range(0, len(opened_files)):
            opened_file = opened_files[i]
            output_las_file = output_las_files[i]
            out_file = laspy.LasData(opened_file.header)
            out_file.points = opened_file.points
            classification = [1 for i in range(0, lengths[i])]  # 1 for non-ground
            for j in range(0, len(ground)):
                if path_flags[ground[j]] == index:
                    classification[ground[j] - accumulated_lengths[index]] = 2
            out_file.classification = classification
            index += 1
            out_file.write(output_las_file)

    def do_filtering_txt(self):
        if self.txt_header is None:
            print(" - ERROR: txt header is None.")
            sys.exit(0)
        header_dic = dict()
        for i in range(0, len(self.txt_header)):
            header_dic[self.txt_header[i]] = i

        index = 0
        lengths = []
        xyz_total = []
        path_flags = []
        for input_file in self.input_files:
            f = open(input_file, 'r')
            single_length = 0
            for line in f:
                arr = line.strip().split(" ")
                xyz_total.append([float(arr[0]), float(arr[1]), float(arr[2])])
                single_length += 1
            lengths.append(single_length)
            path_flags += [index for i in range(single_length)]
            index += 1
            f.close()

        accumulated_lengths = [0]
        for i in range(0, len(lengths)):
            accumulated_lengths.append(accumulated_lengths[i] + lengths[i])

        xyz_total = np.array(xyz_total)

        self.csf.setPointCloud(xyz_total)
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        self.csf.do_filtering(ground, non_ground)

        # 获取output文件，默认保存到原数据所在路径下的csf_filtered文件夹下
        output_dir = os.path.join(os.path.dirname(self.input_files[0]), "csf_filtered")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_txt_files = []
        for input_file_path in self.input_files:
            output_las_file = os.path.join(output_dir, os.path.basename(input_file_path))
            output_txt_files.append(output_las_file)

        index = 0
        for i in range(0, len(self.input_files)):
            outfile = output_txt_files[i]
            classification = [1 for i in range(0, lengths[i])]  # 1 for non-ground
            for j in range(0, len(ground)):
                if path_flags[ground[j]] == index:
                    classification[ground[j] - accumulated_lengths[index]] = 2
            fout = open(outfile, 'w')
            fin = open(self.input_files[i], 'r')
            line_index = 0
            for line in fin:
                arr = line.strip().split(" ")
                arr[header_dic['c']] = str(classification[line_index])
                fout.write(" ".join(arr) + "\n")
                line_index += 1

            fout.close()
            fin.close()
            index += 1
