# coding:utf-8
# 3D voxel LAI inversion from airborne discrete LiDAR data
# author: Jianbo Qi
# date: 2017-8-9

import laspy
import numpy as np
from collections import defaultdict
from enum import Enum
import math


class PulseType(Enum):
    VEG_GROUND = 0
    PURE_VEG = 1
    PURE_GROUND = 2
    NOT_DEFINED = 3


class Pulse:
    def __init__(self):
        self.point_list = []
        self.return_number_list = []
        self.scan_angle_list = []
        self.gps_time_list = []
        self.classification_list = []
        self.intensity_list = []
        self.number_of_return_list = []
        self.pulse_type = PulseType.NOT_DEFINED
        self.pulse_direction = None  # normalized pulse direction

        # calculated value
        self.pulse_incident_intensity = []  # the incident energy at the position of each return

    def merge_pulse(self, pulse):
        self.point_list += pulse.point_list
        self.return_number_list += pulse.return_number_list
        self.scan_angle_list += pulse.scan_angle_list
        self.gps_time_list += pulse.gps_time_list
        self.classification_list += pulse.classification_list
        self.intensity_list += pulse.intensity_list

    def get_cell_coordinates_of_returns(self, grid_size, box_min):
        cell_list = []
        for point in self.point_list:
            cell_list.append(np.floor((np.array(point)-np.array(box_min))/float(grid_size)))
        return cell_list

    def get_point_num(self):
        return len(self.point_list)

    def get_gps_time(self):
        if len(self.gps_time_list) > 0:
            return self.gps_time_list[0]
        print("No gps time")

    def get_scann_angle_rank(self):
        if len(self.scan_angle_list) > 0:
            return self.scan_angle_list[0]
        print("No scan angle")

    def print_string(self):
        print("*******start of pulse*********")
        # print "point_list: ", self.point_list
        print(("number_of_return_list: ", self.number_of_return_list))
        print(("return_number_list: ", self.return_number_list))
        print(("scan_angle_list: ", self.scan_angle_list))
        print(("gps_time_list: ", self.gps_time_list))
        print(("classification_list: ", self.classification_list))
        print(("intensity_list: ", self.intensity_list))
        print("*******end of pulse*********")


def count_points_of_pulse_list(pulse_list):
    total_num = 0
    for pulse in pulse_list:
        total_num += pulse.get_point_num()
    return total_num


def parse_pulses_from_discrete_point_cloud_ext(file_path, fileType, txtHeader):
    in_xyz_totals = []
    in_num_returns = []
    in_return_num = []
    in_gps_time = []
    in_scan_angle_rank = []
    in_classification = []
    in_intensity = []
    in_sourceID = []

    if fileType == "las":
        inFile = laspy.read(file_path)
        in_xyz_totals = inFile.xyz
        in_sourceID = inFile.pt_src_id
        in_num_returns = inFile.num_returns
        in_return_num = inFile.return_num
        in_gps_time = inFile.gps_time
        in_scan_angle_rank = inFile.scan_angle_rank
        in_classification = inFile.classification
        in_intensity = inFile.intensity
    else: # txt
        if txtHeader is None:
            print("Header is needed for txt format")
            import sys
            sys.exit(0)

        headerDic = dict()
        for i in range(0, len(txtHeader)):
            headerDic[txtHeader[i]] = i

        f = open(file_path,'r')
        for line in f:
            arr = line.strip().split(" ")
            in_xyz_totals.append([float(arr[headerDic['x']]),float(arr[headerDic['y']]),float(arr[headerDic['z']])])
            in_classification.append(float(arr[headerDic['c']]))
            in_intensity.append(float(arr[headerDic['i']]))
            in_gps_time.append(float(arr[headerDic['t']]))
            in_return_num.append(float(arr[headerDic['r']]))
            in_num_returns.append(float(arr[headerDic['n']]))
            in_scan_angle_rank.append(float(arr[headerDic['a']]))
            in_sourceID.append(0)
        in_xyz_totals = np.array(in_xyz_totals)
        in_num_returns = np.array(in_num_returns)
        in_return_num = np.array(in_return_num)
        in_gps_time = np.array(in_gps_time)
        in_scan_angle_rank = np.array(in_scan_angle_rank)
        in_classification = np.array(in_classification)
        in_intensity = np.array(in_intensity)
        in_sourceID = np.array(in_sourceID)

    sourceID_unique = set(in_sourceID)

    print(" - Total number of points: ", len(in_xyz_totals))

    final_pulses = []
    error_pulses = []

    for srcID in sourceID_unique:
        xyz_total = in_xyz_totals[in_sourceID == srcID]
        number_of_returns = in_num_returns[in_sourceID == srcID]
        return_number = in_return_num[in_sourceID == srcID]
        gps_time = in_gps_time[in_sourceID == srcID]
        scan_angle_rank = in_scan_angle_rank[in_sourceID == srcID]
        classification = in_classification[in_sourceID == srcID]
        intensity = in_intensity[in_sourceID == srcID]
        D = defaultdict(list)
        for i, item in enumerate(gps_time):
            D[item].append(i)
        D = {k: v for k, v in list(D.items())}
        for key in D:
            pulse = Pulse()
            for id in D[key]:
                pulse.point_list.append(xyz_total[id])
                pulse.return_number_list.append(return_number[id])
                pulse.gps_time_list.append(gps_time[id])
                pulse.classification_list.append(classification[id])
                pulse.intensity_list.append(intensity[id])
                pulse.scan_angle_list.append(scan_angle_rank[id])
                pulse.number_of_return_list.append(number_of_returns[id])
            # validate pulse
            # return number: it should not have duplicated values.
            if len(set(pulse.return_number_list)) != len(pulse.return_number_list):
                error_pulses.append(pulse)
                continue
            # number of return: it should be the same
            if len(set(pulse.number_of_return_list)) != 1:
                error_pulses.append(pulse)
                continue
            # number of return should equal to return number
            if pulse.number_of_return_list[0] != len(pulse.return_number_list):
                error_pulses.append(pulse)
                continue
            # scan angle
            if len(set(pulse.scan_angle_list)) != 1:
                error_pulses.append(pulse)
                continue
            final_pulses.append(pulse)
    errorPoints = count_points_of_pulse_list(error_pulses)
    print(" - Total number of points from incomplete pulses: ", errorPoints, "->", "%.2f"%(100*errorPoints/len(in_xyz_totals)),"%")

    # sort the points in each pulse according to return number
    for pulse in final_pulses:
        pulse.point_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.point_list))]
        pulse.scan_angle_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.scan_angle_list))]
        pulse.classification_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.classification_list))]
        pulse.intensity_list = [x for _, x in sorted(zip(pulse.return_number_list, pulse.intensity_list))]
        pulse.return_number_list = sorted(pulse.return_number_list)

    # determine the type of pulse according to classification
    num_vg, num_g, num_v = 0, 0, 0
    for pulse in final_pulses:
        if 1 in pulse.classification_list and 2 in pulse.classification_list:
            pulse.pulse_type = PulseType.VEG_GROUND
            num_vg += 1
        elif all([x == 2 for x in pulse.classification_list]):
            pulse.pulse_type = PulseType.PURE_GROUND
            num_g += 1
        else:
            pulse.pulse_type = PulseType.PURE_VEG
            num_v += 1

    print(" - Total number of Pulses: ", len(final_pulses))
    print("   - Pure-Vegetation Pulse: ", num_v, " -> %.2f"%(100*num_v/len(final_pulses)),"%")
    print("   - Pure-Ground Pulse: ", num_g, " -> %.2f"%(100*num_g/len(final_pulses)),"%")
    print("   - Vegetation-ground Pulse: ", num_vg, " -> %.2f"%(100*num_vg/len(final_pulses)),"%")

    # determine the pulse direction
    scan_angle_direction_dic = dict()
    for pulse in final_pulses:
        point_num = pulse.get_point_num()
        # directly get pulse direction from the coordinates of the points
        # for one-return pulses, using nearest multiple-return pulses
        if point_num > 1:
            p_dir = np.array(pulse.point_list[point_num - 1] - pulse.point_list[0])
            p_dir /= np.linalg.norm(p_dir)
            pulse.pulse_direction = p_dir
            scan_angle = pulse.get_scann_angle_rank()
            if scan_angle in scan_angle_direction_dic:
                scan_angle_direction_dic[scan_angle].append(pulse.pulse_direction)
            else:
                scan_angle_direction_dic[scan_angle] = [pulse.pulse_direction]
    for scan_angle in scan_angle_direction_dic:
        scan_angle_direction_dic[scan_angle] = sum(scan_angle_direction_dic[scan_angle])\
                                               / float(len(scan_angle_direction_dic[scan_angle]))
        # print(scan_angle, scan_angle_direction_dic[scan_angle])
    pulse_with_no_direction = 0
    for pulse in final_pulses:
        if pulse.get_point_num() == 1:
            if pulse.get_scann_angle_rank() in scan_angle_direction_dic:
                pulse.pulse_direction = scan_angle_direction_dic[pulse.get_scann_angle_rank()]
            else:
                pulse.pulse_direction = np.array([0, 0, -1])
                pulse_with_no_direction+=1
    print(" - Pulse with no direction: ", pulse_with_no_direction, " -> %.2f" % (100 * pulse_with_no_direction / len(final_pulses)), "%")



    # start_p, end_p = estimate_flight_line_on_xy_plane(final_pulses)
    # start_p, end_p = estimate_flight_line_on_xy_plane_ext(final_pulses)
    # start_p2d = np.array([start_p[0], start_p[1]])
    # end_p2d = np.array([end_p[0], end_p[1]])
    # for pulse in final_pulses:
    #     point_num = pulse.get_point_num()
    #     pulse_scan_angle = np.abs(pulse.get_scann_angle_rank())
    #     if point_num > 1:  # directly get pulse direction from the coordinates of the points
    #         p_dir = np.array(pulse.point_list[point_num - 1] - pulse.point_list[0])
    #         p_dir /= np.linalg.norm(p_dir)
    #     else:  # estimation from incident angle
    #         if pulse_scan_angle > 0:
    #             point = pulse.point_list[0]
    #             point2d3 = np.array([point[0], point[1]])
    #             distance, intersect_p = distance_to_line(start_p2d, end_p2d, point2d3)
    #             flight_height = pulse.point_list[0][2] + distance / np.tan(pulse_scan_angle/180.0*np.pi)
    #             p_dir = np.array([intersect_p[0], intersect_p[1], flight_height]) - np.array(point)
    #             p_dir /= np.linalg.norm(p_dir)
    #         else:
    #             p_dir = np.array([0, 0, -1])
    #     pulse.pulse_direction = p_dir
    return final_pulses


# calculate distance between point2d3 and the line defined with vector
def distance_to_line(point2d1, point2d2, point2d3):
    p1p2 = point2d2 - point2d1
    p1p3 = point2d3 - point2d1
    t = np.dot(p1p2, p1p3)/np.linalg.norm(p1p2)
    intersected_p = point2d1 + p1p2/np.linalg.norm(p1p2)*t
    distance = np.linalg.norm(point2d3 - intersected_p)
    return distance, intersected_p


# estimation of flight line on the XY plane
def estimate_flight_line_on_xy_plane_ext(_final_pulses):
    scan_angle_azimuth = dict()
    for pulse in _final_pulses:
        point_num = pulse.get_point_num()
        scan_angle = pulse.get_scann_angle_rank()
        if point_num > 3 and scan_angle == -15:
            p1 = pulse.point_list[0]
            p2 = pulse.point_list[point_num - 1]
            p1p2 = p2-p1
            print(p1p2/np.linalg.norm(p1p2))
            p1p2_xy = np.array([p1p2[0], p1p2[1]])
            p1p2_xy /= np.linalg.norm(p1p2_xy)
            vertical = np.array([0, 1])
            angle = math.acos(np.dot(p1p2_xy, vertical))/math.pi*180
            if scan_angle in scan_angle_azimuth:
                scan_angle_azimuth[scan_angle].append(angle)
            else:
                scan_angle_azimuth[scan_angle] = [angle]
            # print scan_angle, angle
    for scan_angle in scan_angle_azimuth:
        print(scan_angle, sum(scan_angle_azimuth[scan_angle])/float(len(scan_angle_azimuth[scan_angle])))


# estimation of flight line on the XY plane
def estimate_flight_line_on_xy_plane(_final_pulses):
    center_gps_time = []
    center_points = []
    points_indice = []
    point_index = 0
    for pulse in _final_pulses:
        scan_angle = pulse.get_scann_angle_rank()
        if scan_angle == 0:
            for i in range(0, pulse.get_point_num()):
                center_gps_time.append(pulse.gps_time_list[i])
                center_points.append(pulse.point_list[i])
                points_indice.append(point_index)
                point_index += 1

    # sorting according to the gps time
    sorted_indice = [x for _, x in sorted(zip(center_gps_time, points_indice))]
    # get the first 50 points and last 50 points
    num = 50
    first_50 = [center_points[i] for i in sorted_indice[:num]]
    last_50 = [center_points[i] for i in sorted_indice[-num:]]
    start_point = np.mean(np.array(first_50), axis=0)
    end_point = np.mean(np.array(last_50), axis=0)
    return start_point, end_point


if __name__ == "__main__":
    pass
    # the input las file should be classified in advance, classification=1 vegetation classification=2 ground
    # all_pulses = parse_pulses_from_discrete_point_cloud_ext
    # (r"E:\Research\23-DART-DAO\simulation\Voxel3DInversion\single_path1_classified.las")
    # f = open(r"E:\Research\23-DART-DAO\simulation\Voxel3DInversion\veg_ground_points.txt", 'w')
    # pulse_index = 0
    # for pulse in all_pulses:
    #     pulse_index += 1
    #     if pulse.pulse_type == PulseType.PURE_GROUND:
    #         point_num = pulse.get_point_num()
    #         if point_num > 0:
    #             for i in range(0, point_num):
    #                 point = pulse.point_list[i]
    #                 f.write("%.3f %.3f %.3f %d %.3f\n" % (point[0], point[1], point[2], pulse_index, pulse.intensity_list[i]))
    # f.close()
