# coding: utf-8
import numpy as np


class BoundingBox:
    def __init__(self, arr):
        self.min_x = arr[0]
        self.min_y = arr[1]
        self.min_z = arr[2]
        self.width = arr[3]
        self.height = arr[4]
        self.depth = arr[5]
        self.num_w = arr[6]
        self.num_h = arr[7]
        self.num_d = arr[8]
        self.max_x = arr[9]
        self.max_y = arr[10]
        self.max_z = arr[11]

    def to_string(self):
        print("minx:", self.min_x)
        print("min_y:", self.min_y)
        print("min_z:", self.min_z)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)
        print("max_z:", self.max_z)


# implementation of a ray traversal algorithm
# input: ray start, ray end, grid size
# returns all the traversed voxels and the corresponding path length
def ray_voxel_traversal(z_max, ray_o, ray_dir, ray_end, grid_size):
    # first, do ray-plane intersection, find the points at the top of the canopy
    t0 = (z_max - ray_o[2]) / ray_dir[2] - 0.000001
    top_p = ray_o + t0 * ray_dir
    return ray_traversal(top_p, ray_end, grid_size)


# implementation of a ray traversal algorithm
# input: ray start, ray end, grid size
# returns all the traversed voxels and the corresponding path length
def ray_traversal(ray_start, ray_end, grid_size):
    visited_voxels = []
    traversed_lengths = []
    # determine the first voxel
    current_voxel = np.floor(ray_start / grid_size)
    start_voxel = np.copy(current_voxel)
    last_voxel = np.floor(ray_end / grid_size)

    dist = np.linalg.norm(ray_start-ray_end)

    # normalized ray direction
    dir = ray_end - ray_start
    dir = dir / np.linalg.norm(dir)

    drcp = np.zeros(3, )
    for i in range(0, 3):
        drcp[i] = 1 / dir[i] if dir[i] != 0 else np.inf

    # In which direction the voxel ids are incremented.
    stepX = np.sign(dir[0])
    stepY = np.sign(dir[1])
    stepZ = np.sign(dir[2])

    # define tMaxX, tMaxY, tMaxZ
    next_voxel_boundary_x = (current_voxel[0]) * grid_size if dir[0] < 0 else (current_voxel[0] + 1) * grid_size
    next_voxel_boundary_y = (current_voxel[1]) * grid_size if dir[1] < 0 else (current_voxel[1] + 1) * grid_size
    next_voxel_boundary_z = (current_voxel[2]) * grid_size if dir[2] < 0 else (current_voxel[2] + 1) * grid_size
    tMaxX = (next_voxel_boundary_x - ray_start[0]) * drcp[0] if dir[0] != 0 else np.inf
    tMaxY = (next_voxel_boundary_y - ray_start[1]) * drcp[1] if dir[1] != 0 else np.inf
    tMaxZ = (next_voxel_boundary_z - ray_start[2]) * drcp[2] if dir[2] != 0 else np.inf

    tDeltaX = abs(grid_size * drcp[0]) if dir[0] != 0 else np.inf
    tDeltaY = abs(grid_size * drcp[1]) if dir[1] != 0 else np.inf
    tDeltaZ = abs(grid_size * drcp[2]) if dir[2] != 0 else np.inf

    t = min(tMaxX, tMaxY, tMaxZ)
    preT = t

    traversed_lengths.append(t)
    visited_voxels.append(np.array(current_voxel))

    while not np.array_equal(current_voxel, last_voxel):
        if tMaxX <= tMaxY:
            if tMaxX < tMaxZ:
                current_voxel[0] += stepX
                tMaxX += tDeltaX
            else:
                current_voxel[2] += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                current_voxel[1] += stepY
                tMaxY += tDeltaY
            else:
                current_voxel[2] += stepZ
                tMaxZ += tDeltaZ
        t = min(tMaxX, tMaxY, tMaxZ)
        if t > dist:
            t = dist
        traversed_lengths.append(t - preT)
        preT = t
        visited_voxels.append(np.array(current_voxel))
        # if len(visited_voxels) > 100:
        #     print("break: ")
        #     print("start-last-voxel:",start_voxel, last_voxel)
        #     print("tranversal: ", visited_voxels)
        #     print("---------------------------------")
        #     break
    # if len(visited_voxels) > 100:
    #     print("start-last-voxel:",start_voxel, last_voxel)
    #     print("len: ", len(visited_voxels))
    #     print("tranversal: ", visited_voxels)
    #     print("---------------------------------")
    return visited_voxels, traversed_lengths


# Implementation of a ray-box intersection algorightm
# This is used to do small path length correction
def ray_box_intersect(ray_start, ray_dir, box_center, grid_size):
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    drcp = np.zeros(3, )
    for i in range(0, 3):
        drcp[i] = 1 / ray_dir[i] if ray_dir[i] != 0 else np.inf
    tmin = -np.inf
    tmax = np.inf
    for i in range(0, 3):
        origin = ray_start[i]
        minVal = box_center[i]-grid_size*0.5
        maxVal = box_center[i]+grid_size*0.5
        if ray_dir[i] == 0:
            if origin < minVal or origin > maxVal:
                return None
        else:
            t1 = (minVal - origin)*drcp[i]
            t2 = (maxVal - origin)*drcp[i]
            if t1 > t2:
                t1,t2 = t2, t1
            tmin = max(t1, tmin)
            tmax = min(t2, tmax)
    return tmax-tmin


def transmittance2lad_simple(T, d):
    # assuming the leaf angle distribution is spherical, and ingore the scanning angle
    G = 0.5
    LAD = -np.log(T) / (G*d)
    return LAD
