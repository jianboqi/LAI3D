# coding: utf-8
from AlphaPADEstimatorBase import AlphaPADEstimatorBase
from multiprocessing import Process, Pool
import os
from Pulse import *
from Utils import *
from sklearn.neighbors import NearestNeighbors
import matlab.engine


class AlphashapePADEstimatorByRaytracing(AlphaPADEstimatorBase):
    def __init__(self, filtered_input_files, file_type, txt_header=None):
        super().__init__(filtered_input_files, file_type, txt_header)
        self.RAY_O_Z = 1000
        self.eng = matlab.engine.start_matlab()

    def pad_inversion(self, parallel_computing=False):
        # First, read all data from all paths
        # all_points = self.read_merged_points()
        # self.alpha_shape_seg(all_points)

        if parallel_computing:
            print(" - ****************Start to process******************")
            pool = Pool(os.cpu_count())
            pool.map(self.lai_inversion_per_flight_path, self.filtered_input_files)
        else:
            for input_file in self.filtered_input_files:
                print(" - **********************************")
                print(" - Processing file: ", input_file)
                self.lai_inversion_per_flight_path(input_file)

    def alpha_shape_seg(self, all_points):
        # seg using matlab
        pass

    def lai_inversion_per_flight_path(self, flight_path_file):
        self.all_pulses = parse_pulses_from_discrete_point_cloud_ext(flight_path_file, self.file_type, self.txt_header)
        self.calculate_incident_energy_of_each_return()
        non_ground_pulse_incident_energy, nbrs_non_ground = self.get_non_ground_pulse_nearest_neighbor_tree()
        print(" - Start ray traversal...")
        total_processed_cells = 0
        for pulse in self.all_pulses:
            total_processed_cells += 1
            if total_processed_cells % 20000 == 0:
                print("   - Processed Pulses: ", total_processed_cells)

    def get_non_ground_pulse_nearest_neighbor_tree(self):
        non_ground_pulse_incident_energy = []
        non_ground_pulse_top_point_array = []
        for pulse in self.all_pulses:
            if pulse.pulse_type != PulseType.PURE_GROUND:
                ray_o = pulse.point_list[0] + self.RAY_O_Z * (-pulse.pulse_direction) - np.array(
                    [self.bb.min_x, self.bb.min_y, self.bb.min_z])
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
        pure_ground_points_array = np.array(pure_ground_points_array) - np.array(
            [self.bb.min_x, self.bb.min_y, self.bb.min_z])
        # using x y of pure ground point to build a nearest search tree
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pure_ground_points_array[:, 0:2])

        number_of_invalided_ground_veg_points = 0

        for pulse in self.all_pulses:
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

    def read_merged_points(self):
        all_points = None
        for filtered_input_file in self.filtered_input_files:
            if self.file_type == "las":
                in_file = laspy.read(filtered_input_file)
                in_xyz_totals = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()
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
                        [float(arr[headerDic['x']]), float(arr[headerDic['y']]), float(arr[headerDic['z']])])
                in_xyz_totals = np.array(in_xyz_totals)
                if all_points is None:
                    all_points = in_xyz_totals
                else:
                    all_points = np.vstack((all_points, in_xyz_totals))
        return all_points
