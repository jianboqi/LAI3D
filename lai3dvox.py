# coding: utf-8
# author: Jianbo QI
# date: 2019-6-26
# 从机载激光雷达数据中反演三维叶面积体密度
# Invert plant area density from airborne lidar data

import argparse
import time
import glob
import os


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-i", nargs='*', help="Input file.", required=True)
    parse.add_argument("-resolution", help="Resolution of voxel.", required=True, type=float, default=1)
    parse.add_argument("-f", required=True, type=str, default="las", help="文件类型，可以是txt，las格式")

    # 如果文件类型是txt格式（第一行为文件头，必须，如果实在没有，就随便写点啥吧，比如“你好啊”，反正也不重要），
    # 请提供每一列的具体含义，x-代表x坐标，y-代表y坐标，z-代表z坐标，c-分类，i-强度，t-GPS时间，r-回波序号
    # n-每个脉冲总回波次数，a-扫描角度，o-其他无关列均用o代替
    # if filetype if txt (-f txt), the txt file must have a header line, the content is not important
    # -header specify the name of each column, e.g., -header xyzcitrnao, x-x coordinate, y-y coordinate,
    # z-z coordinate, c-classification, i-intensity, t-gps time, r-return number, n-number returns,a-scan angle rank,
    # o-other columns
    parse.add_argument("-header", help="If file type is txt, specify the header meanings", type=str)

    # Estimator type: VoxByRtIntensity, VoxByColumnNumber
    parse.add_argument("-estimator", required=True, help="PAD estimator", type=str, default="VoxByColumnNumber")

    # 是否执行CSF滤波，最终结果中1代表非地面点，2代表地面点
    # whether csf ground filtering is applied, if you specify -csf, then original point cloud will be filtered,
    # the result file will be stored into a folder named csf_filtered, in the filtered file, classification=1 stands
    # for non-ground points, classification=2 means ground points
    parse.add_argument("-csf", help="Whether csf ground filtering is applied", action='store_true')
    parse.add_argument("-csfClothRigidness", help="Rigidness parameter for CSF，1 for steep slope，2 for relief，"
                                                  "3 for urban area", type=int, default=1)

    # 是否要进行并行计算，将会使用计算全部核进行计算
    parse.add_argument("-para", help="Will parallel computing is enabled, default False", action='store_true',
                       default=False)

    # 输出结果 保存为numpy的npy格式，该格式可以直接通过np.load进行读取，便于进一步处理
    parse.add_argument("-o", type=str, help="Output 3D PAD path in npy format", required=True)

    args = parse.parse_args()

    start = time.perf_counter()  # 开始记录时间

    # #######################开始处理输入参数##################################
    # 将所以输入文件存在一个数组中
    # 这里为什么要用glob.glob呢？ 因为它可以匹配任意字符串，比如当你命令行为 python 3DLidarInversion.py -i input*.las时,
    # 可以匹配目录下所有以input开头，las结束的文件，e.g., input1.las, input2.las, input3.las...
    input_files = []
    for input_file in args.i:
        input_files += glob.glob(input_file)

    resolution = args.resolution  # 体元分辨率
    print(" - Resolution: ", resolution)

    fileType = args.f  # 文件类型
    print(" - File Type: ", fileType)
    txtHeader = None
    if fileType == "txt":  # 如果是文本文件，则需要提供文件头
        txtHeader = args.header

    output_pad_file_path = args.o

    filtered_output_files = input_files  # 滤波后文件保存路径

    hasCSF = args.csf
    csfClothRigidness = 1
    if hasCSF is True:
        print(" - Include ground filtering: ", hasCSF)
        csfClothRigidness = args.csfClothRigidness
        from GroundFilter import GroundFilter
        gf = GroundFilter(csfClothRigidness, input_files, fileType, txtHeader)
        print(" - Start to do ground filtering: ", hasCSF)
        gf.do_filtering()

        filtered_output_files = []
        filtered_dir = os.path.join(os.path.dirname(input_files[0]), "csf_filtered")
        for input_file_path in input_files:
            output_las_file = os.path.join(filtered_dir, os.path.basename(input_file_path))
            filtered_output_files.append(output_las_file)
    # #######################处理输入参数结束#######################################
    # #######################好了，数据已经准备好了，开始反演#########################
    pad_estimator = None
    print(" - Using PAD Estimator:", args.estimator)
    if args.estimator == "VoxByRtIntensity":
        from VoxelPADEstimatorByRaytracing import VoxelPADEstimatorByRaytracing
        pad_estimator = VoxelPADEstimatorByRaytracing(filtered_output_files, fileType, resolution, txtHeader)
    elif args.estimator == "VoxByColumnNumber":
        from VoxelPADEstimatorByColumnPoint import VoxelPADEstimatorByColumnPoint
        pad_estimator = VoxelPADEstimatorByColumnPoint(filtered_output_files, fileType, resolution, txtHeader)
    pad_estimator.pad_inversion(args.para)
    pad_estimator.save_pad(output_pad_file_path)
    # #######################反演结束#############################

    end = time.process_time()  # 结束记录时间
    print(" - Time: ", "%.3fs" % (end - start))
    # #######################到这里就结束了,谢谢观看,打扰了################################
