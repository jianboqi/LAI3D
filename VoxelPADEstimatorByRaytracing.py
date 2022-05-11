# coding: utf-8
# author: Jianbo QI
# date: 2019-6-26
# 估算PAD，从激光点云数据中

import os
from Pulse import *
from Utils import *
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Process, Pool
from VoxelPADEstimatorBase import VoxelPADEstimatorBase


class VoxelPADEstimatorByRaytracing(VoxelPADEstimatorBase):
    def __init__(self, filtered_input_files, file_type, resolution, txt_header=None):
        super().__init__(filtered_input_files, file_type, resolution, txt_header)

        self.RAY_O_Z = 1000
        # rows cols heights 每个体元入射能量
        self.out_incident_energy = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d), dtype=float)
        # rows cols heights 每个体元透过的能量
        self.out_transmitted_energy = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d), dtype=float)
        # record how many rays has traversed a voxel
        self.out_number_of_traversed_rays = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d))
        # record the averaged path length that has traversed a voxel
        self.out_path_lengths = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d), dtype=float)

    def pad_inversion(self, parallel_computing=False):
        if parallel_computing:
            print(" - ****************Start to process******************")
            pool = Pool(os.cpu_count())
            pool.map(self.lai_inversion_per_flight_path, self.filtered_input_files)
        else:
            for input_file in self.filtered_input_files:
                print(" - **********************************")
                print(" - Processing file: ", input_file)
                self.lai_inversion_per_flight_path(input_file)

        # 计算PAD
        total_pai = 0
        self.pad = np.zeros((self.bb.num_h, self.bb.num_w, self.bb.num_d))
        for i in range(0, self.bb.num_h):
            for j in range(0, self.bb.num_w):
                for k in range(0, self.bb.num_d):
                    incident = self.out_incident_energy[i][j][k]
                    out_energy = self.out_transmitted_energy[i][j][k]
                    tot_length = self.out_path_lengths[i][j][k]
                    num_paths = self.out_number_of_traversed_rays[i][j][k]
                    if incident != 0 and out_energy != 0:
                        self.pad[i][j][k] = transmittance2lad_simple(out_energy / incident, tot_length / float(num_paths))
                        total_pai += self.pad[i][j][k] * self.resolution * self.resolution * self.resolution

        print(" - Plot LAI: ", total_pai / float(self.bb.num_w * self.bb.num_h * self.resolution * self.resolution))

    def lai_inversion_per_flight_path(self, flight_path_file):
        self.all_pulses = parse_pulses_from_discrete_point_cloud_ext(flight_path_file, self.file_type, self.txt_header)
        print(" - Matrix dimensions: ", self.bb.num_w, self.bb.num_h, self.bb.num_d)
        self.calculate_incident_energy_of_each_return()
        non_ground_pulse_incident_energy, nbrs_non_ground = self.get_non_ground_pulse_nearest_neighbor_tree()
        print(" - Start ray traversal...")
        total_processed_cells = 0
        for pulse in self.all_pulses:
            total_processed_cells += 1
            if total_processed_cells % 20000 == 0:
                print("   - Processed Pulses: ", total_processed_cells)

            box_min = [self.bb.min_x, self.bb.min_y, self.bb.min_z]
            voxel_dimentions = [self.bb.num_w, self.bb.num_h, self.bb.num_d]
            ray_o = pulse.point_list[0] + self.RAY_O_Z * (-pulse.pulse_direction) - np.array(box_min)
            ray_end = pulse.point_list[pulse.get_point_num() - 1] - np.array(box_min)
            visited_voxels, traversed_lengths = ray_voxel_traversal(self.bb.max_z-self.bb.min_z, ray_o,
                                                                    pulse.pulse_direction, ray_end, self.resolution)
            cellIndex = 0
            Lc = 0
            for visited_cell in visited_voxels:
                x, y, z = int(visited_cell[0]), int(visited_cell[1]), int(visited_cell[2])
                if not ((-1 < x < voxel_dimentions[0]) and (-1 < y < voxel_dimentions[1]) and (
                        -1 < z < voxel_dimentions[2])):
                    continue
                self.out_number_of_traversed_rays[voxel_dimentions[1] - y - 1][x][z] += 1
                if cellIndex == 0:
                    xc, yc, zc = (x + 0.5) * self.resolution, (y + 0.5) * self.resolution, (z + 0.5) * self.resolution
                    Lc = ray_box_intersect(np.array([xc, yc, zc]), pulse.pulse_direction, np.array([xc, yc, zc]),
                                           self.resolution)
                self.out_path_lengths[voxel_dimentions[1] - y - 1][x][z] += Lc
                cellIndex += 1

            pulse_point_cell_list = pulse.get_cell_coordinates_of_returns(self.resolution, box_min)
            incident_energy_list = pulse.pulse_incident_intensity

            # remove multi points in the same cell
            tmp_cell_list = [pulse_point_cell_list[0]]
            tmp_intensity_list = [incident_energy_list[0]]
            if pulse.get_point_num() >= 2:
                for i in range(1, len(pulse_point_cell_list)):
                    if not (pulse_point_cell_list[i] == pulse_point_cell_list[i - 1]).all():
                        tmp_cell_list.append(pulse_point_cell_list[i])
                        tmp_intensity_list.append(incident_energy_list[i])
            pulse_point_cell_list = tmp_cell_list
            incident_energy_list = tmp_intensity_list

            corrected_return_num = len(incident_energy_list)

            # Calculate incident energy and transmitted energy for each cell
            cur_return_cell_index = 0
            if pulse.pulse_type == PulseType.VEG_GROUND:
                for i in range(0, len(visited_voxels) - 1):
                    visited_cell = visited_voxels[i]
                    x, y, z = int(visited_cell[0]), int(visited_cell[1]), int(visited_cell[2])
                    if not ((-1 < x < voxel_dimentions[0]) and (-1 < y < voxel_dimentions[1]) and (
                            -1 < z < voxel_dimentions[2])):
                        continue

                    try:
                        if (visited_cell == pulse_point_cell_list[cur_return_cell_index]).all():
                            self.out_incident_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index]
                            self.out_transmitted_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index + 1]

                            cur_return_cell_index += 1
                        else:
                            self.out_incident_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index]
                            self.out_transmitted_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index]
                    except:
                        print("Exception")

            if pulse.pulse_type == PulseType.PURE_VEG:
                for visited_cell in visited_voxels:
                    x, y, z = int(visited_cell[0]), int(visited_cell[1]), int(visited_cell[2])
                    if not (-1 < x < voxel_dimentions[0] and -1 < y < voxel_dimentions[1] and -1 < z < voxel_dimentions[
                        2]):
                        continue
                    try:
                        if (visited_cell == pulse_point_cell_list[cur_return_cell_index]).all():
                            self.out_incident_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index]
                            if cur_return_cell_index < corrected_return_num - 1:
                                self.out_transmitted_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                    cur_return_cell_index + 1]

                                if np.array_equal(visited_cell, np.array([10, 10, 28])):
                                    print(self.out_transmitted_energy[voxel_dimentions[1] - y - 1][x][z] /
                                          self.out_incident_energy[voxel_dimentions[1] - y - 1][x][z])

                            cur_return_cell_index += 1
                        else:
                            self.out_incident_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index]
                            self.out_transmitted_energy[voxel_dimentions[1] - y - 1][x][z] += incident_energy_list[
                                cur_return_cell_index]
                    except:
                        print("Exception")

            if pulse.pulse_type == PulseType.PURE_GROUND:
                ray_o = pulse.point_list[0] + self.RAY_O_Z * (-pulse.pulse_direction) - np.array(box_min)
                t0 = (self.bb.max_z - ray_o[2]) / pulse.pulse_direction[2]
                top_p = ray_o + t0 * pulse.pulse_direction - 0.0001
                distances, indices = nbrs_non_ground.kneighbors([top_p[0:2]])
                for i in range(0, len(visited_voxels) - 1):
                    visited_cell = visited_voxels[i]
                    x, y, z = int(visited_cell[0]), int(visited_cell[1]), int(visited_cell[2])
                    if not (-1 < x < voxel_dimentions[0] and -1 < y < voxel_dimentions[1] and -1 < z < voxel_dimentions[
                        2]):
                        continue
                    self.out_incident_energy[voxel_dimentions[1] - y - 1][x][z] += non_ground_pulse_incident_energy[
                        indices[0][0]]
                    self.out_transmitted_energy[voxel_dimentions[1] - y - 1][x][z] += non_ground_pulse_incident_energy[
                        indices[0][0]]

    def get_non_ground_pulse_nearest_neighbor_tree(self):
        non_ground_pulse_incident_energy = []
        non_ground_pulse_top_point_array = []
        for pulse in self.all_pulses:
            if pulse.pulse_type != PulseType.PURE_GROUND:
                ray_o = pulse.point_list[0] + self.RAY_O_Z * (-pulse.pulse_direction) - np.array([self.bb.min_x, self.bb.min_y, self.bb.min_z])
                t0 = (self.bb.max_z - ray_o[2]) / pulse.pulse_direction[2]
                top_p = ray_o + t0 * pulse.pulse_direction - 0.0001
                non_ground_pulse_top_point_array.append(top_p)
                non_ground_pulse_incident_energy.append(pulse.pulse_incident_intensity[0])
        non_ground_pulse_top_point_array = np.array(non_ground_pulse_top_point_array)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(non_ground_pulse_top_point_array[:, 0:2])
        return non_ground_pulse_incident_energy, nbrs

    def calculate_incident_energy_of_each_return(self):
        # find pure ground pulse to build nearest search data structure
        # pulse_type == PURE_GROUND
        pure_ground_points_array = []
        pure_ground_intensity_array = []
        for pulse in self.all_pulses:
            if pulse.pulse_type == PulseType.PURE_GROUND:
                pure_ground_points_array += pulse.point_list
                pure_ground_intensity_array += pulse.intensity_list
        pure_ground_points_array = np.array(pure_ground_points_array)  # - np.array([self.bb.min_x, self.bb.min_y, self.bb.min_z])
        # using x y of pure ground point to build a nearest search tree
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pure_ground_points_array[:, 0:2])

        number_of_invalided_ground_veg_points = 0

        for pulse in self.all_pulses:

            pulse_point_cell_list = pulse.get_cell_coordinates_of_returns(self.resolution, [self.bb.min_x, self.bb.min_y, self.bb.min_z])
            pulse_intensity_list = pulse.intensity_list
            point_num = pulse.get_point_num()

            # calculate incident energy at return position
            # for vegetation-ground pulse, use the nearest ground pulse to correct ground reflectance
            incident_energy_list = [0 for i in range(0, point_num)]
            if pulse.pulse_type == PulseType.VEG_GROUND:
                # distances, indices = nbrs.kneighbors([pulse_point_cell_list[point_num - 1][0:2]])
                distances, indices = nbrs.kneighbors([pulse.point_list[point_num - 1][0:2]])
                qgstart = pure_ground_intensity_array[indices[0][0]]  # near_pure_ground_intensity
                qg = pulse_intensity_list[point_num - 1]  # last_ground_intensity

                if qgstart <= qg:
                    number_of_invalided_ground_veg_points += 1
                    for i in range(0, point_num):
                        incident_energy_list[i] = sum(pulse_intensity_list[i:])

                else:
                    for i in range(0, point_num):
                        incident_energy_list[i] = qg * sum(pulse_intensity_list[0:i]) + \
                                                  qgstart * sum(pulse_intensity_list[i: point_num - 1])
                        incident_energy_list[i] /= (qgstart - qg)
            if pulse.pulse_type == PulseType.PURE_VEG:
                for i in range(0, point_num):
                    incident_energy_list[i] = sum(pulse_intensity_list[i:])
            pulse.pulse_incident_intensity = incident_energy_list

        print(" - Invalid Ground Points: ", number_of_invalided_ground_veg_points, " -> ",
              "%.2f" % (100 * number_of_invalided_ground_veg_points / count_points_of_pulse_list(self.all_pulses)), "%")

