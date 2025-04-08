import torch
from torch import nn
import numpy as np

from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.decomposition import PCA
from tqdm import tqdm
import open3d as o3d
from utils.graphics_utils import fov2focal

def random_size_offset(flake_number, factor=0.3):
    size_range = torch.rand([flake_number, 3], device="cuda")
    return (torch.cat((size_range, size_range, size_range), dim=0) - 0.5) * factor

def gen_snowfall_vector(flake_number, fall_speed = 1.5, gravity=None):
    snow_fall = np.random.rand(flake_number, 3) * fall_speed
    snow_fall[:, 0] = (snow_fall[:, 2] ) * -0.4
    snow_fall[:, 2] = (snow_fall[:, 2] ) * -0.4
    if gravity is not None:
        default_gravity = np.array([0, -1, 0], dtype=np.float32)
        gravity = gravity / np.linalg.norm(gravity)
        
        axis = np.cross(default_gravity, gravity)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(default_gravity, gravity), -1.0, 1.0))
            
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            ux, uy, uz = axis
            rotation_matrix = np.array([
                [cos_angle + ux * ux * (1 - cos_angle),
                    ux * uy * (1 - cos_angle) - uz * sin_angle,
                    ux * uz * (1 - cos_angle) + uy * sin_angle],
                [uy * ux * (1 - cos_angle) + uz * sin_angle,
                    cos_angle + uy * uy * (1 - cos_angle),
                    uy * uz * (1 - cos_angle) - ux * sin_angle],
                [uz * ux * (1 - cos_angle) - uy * sin_angle,
                    uz * uy * (1 - cos_angle) + ux * sin_angle,
                    cos_angle + uz * uz * (1 - cos_angle)]
            ], dtype=np.float32)
            
            snow_fall = np.dot(snow_fall, rotation_matrix.T)
    snow_fall = torch.from_numpy(snow_fall).to("cuda")
    snow_fall = torch.cat((snow_fall, snow_fall, snow_fall), dim=0)
    return snow_fall

def filter_outlier(points_position, k=20, threshold=0.5):
    ori_num = points_position.shape[0]
    points = points_position.detach().cpu().numpy()
    neighbors = NearestNeighbors(n_neighbors=k, radius=threshold, algorithm='auto', n_jobs=-1).fit(points)
    distances, indices = neighbors.kneighbors(points)
    mean_distances = np.mean(distances, axis=1)
    mask = mean_distances < threshold
    mask = torch.from_numpy(mask).bool()
    filtered_points = points_position[mask]
    print(ori_num-filtered_points.shape[0], " outlier points filtered.")
    return filtered_points, mask

def densification_snow(points_position, k=7, min_threshold=0.01, max_threshold=0.06, g_vec=np.array([0.0, -1.0, 0.0]), limit=None, angle_threshold=0.2):
    points = points_position.detach().cpu().numpy()
    print("before densification: ", points.shape[0])
    neighbors = NearestNeighbors(n_neighbors=k, radius=max_threshold, algorithm='auto', n_jobs=-1).fit(points)
    distances, indices = neighbors.kneighbors(points)
    mean_distances = np.mean(distances, axis=1)
    densified_points = points_position

    for idx, n_idx in enumerate(tqdm(indices,desc="kn densification..")):
        if limit is not None:
            if densified_points.shape[0] > limit:
                print("Reach limit, after densification: ", densified_points.shape[0])
                return densified_points
        center = points[idx]
        n_idx = np.random.choice(n_idx)
        init_neighbours = points[n_idx]

        n_direction = init_neighbours - center
        res_dot = np.abs(np.matmul(n_direction/np.linalg.norm(n_direction), g_vec))
        if res_dot >angle_threshold:
            continue
        dist = distances[idx,-1]
        if dist>max_threshold or dist < min_threshold:
            continue
        point_clone = n_direction/3.0 + center
        point_clone = np.array([point_clone])
        densified_points = torch.cat([densified_points, torch.from_numpy(point_clone).float().to("cuda")], axis=0)

    print("after densification: ", densified_points.shape[0])
    return densified_points

def init_scale(base_position, env_points, env_scale, base_scale=-5.0):
    base_position_clone = base_position.clone().detach().cpu().numpy()
    res_scale = np.ones(base_position_clone.shape, dtype=np.float32) * base_scale
    env_points_clone = env_points.clone().detach().cpu().numpy()
    env_scale_clone = env_scale.clone().detach().cpu().numpy()
    count=0

    neighbors = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=1).fit(env_points_clone)
    _, indices = neighbors.kneighbors(base_position_clone)
    print("Initiating scales...")
    for idx, n_idx in enumerate(indices):
    # for idx, n_idx in enumerate(tqdm(indices, desc="initiating scales...")):
        scale = np.exp(env_scale_clone[n_idx])
        # print(np.mean(scale))
        if np.mean(scale) < 1.0:
            continue
        res_scale[idx][0] = base_scale+0.5
        res_scale[idx][1] = base_scale
        res_scale[idx][2] = base_scale+0.5
        count+=1

    res_scale = torch.from_numpy(res_scale).float().to("cuda")
    print("number of bigger scales: ", count)
    return res_scale

def np_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def np_inv_sigmoid(x):
    return np.log(x/(1-x))

def estimate_gravity_direction(point_cloud):
    points_np = point_cloud.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(points_np)
    gravity_direction_np = pca.components_[-1]

    if gravity_direction_np[1] > 0:
        gravity_direction_np = -gravity_direction_np

    gravity_direction = gravity_direction_np / np.linalg.norm(gravity_direction_np)
    
    return gravity_direction

def compute_average_distance(base_position, n_neighbors=10, algorithm='auto'):
    points = base_position.clone().detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(points)
    distances, _ = nbrs.kneighbors(points)

    mean_distances = np.mean(distances[:, 1:], axis=1)
    overall_average = np.mean(mean_distances)

    return overall_average, mean_distances

def local_plane_densification(base_position, g_vec=None, radius=0.15, samples_per_plane=5, iteration=1, algorithm='auto', noise_intensity=0.05):
    """
    Desification based on fitting local plane
    Args:
        points (numpy.ndarray): (N, 3)
        radius (float): neighbour radius
        samples_per_plane (int): interpolate numbers
    Returns:
        numpy.ndarray: output desified point cloud
    """

    points = base_position.clone().detach().cpu().numpy()
    norm = np.linalg.norm(points, axis=1)

    
    points_map = None
    if g_vec is None:
        g_vec=np.array([0.0, -1.0, 0.0])
    for iter in range(iteration):
        dense_points = []
        f = (iteration-(iter))/iteration
        radius_ex = radius*f
        if points_map is None:
            points_map = points
        nbrs = NearestNeighbors(n_neighbors=5, algorithm=algorithm,radius=radius_ex).fit(points_map)
        for i in tqdm(range(len(points)), desc="Local plane densification processing for iteration "+str(iter+1)):
            distances, indices = nbrs.radius_neighbors([points[i]])
            local_points = points_map[indices[0]]
            if len(local_points) < 3:
                continue 
            
            std_dist = np.std(distances[0])
            median_dist = np.nanmedian(distances[0])
            
            # Fit the plane
            centroid = local_points.mean(axis=0)
            local_points_centered = local_points - centroid
            _, _, vh = np.linalg.svd(local_points_centered)
            normal = vh[-1]
            if np.abs(np.matmul(normal, g_vec)) < 0.7:
                continue
            
            # Random interpolation
            for _ in range(samples_per_plane * (iter+1)):
                noise = np.random.uniform(-noise_intensity, noise_intensity, size=3)
                random_point = centroid + np.random.uniform(-median_dist/(std_dist*2+1.0), median_dist/(std_dist*2+1.0), size=3)
                random_point -= np.dot(random_point - centroid, normal) * normal + noise
                dense_points.append(random_point)
        points_map = np.vstack([points_map, np.array(dense_points)])
    output = torch.from_numpy(points_map).to("cuda").float()
    
    return output



import math

def depth_to_point_cloud(camera, ref_depth):
    """
    Initialize snow points based on depth reprojection
    """
    device = ref_depth.device
    H, W = camera.image_height, camera.image_width
    f_x = fov2focal(camera.FoVx, W)
    f_y = fov2focal(camera.FoVy, H)
    c_x = W * 0.5
    c_y = H * 0.5

    u = torch.arange(0, W, device=device, dtype=ref_depth.dtype)
    v = torch.arange(0, H, device=device, dtype=ref_depth.dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # grid_v, grid_u: [H, W]

    depth = ref_depth[0]  # [H, W]

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = camera.camera_center.to(device)
    offset_z = torch.abs(cam_center[2])
    corrected_depth = depth - 1 * offset_z
    corrected_depth = depth

    x_cam = (grid_u - c_x) * corrected_depth / f_x
    y_cam = (grid_v - c_y) * corrected_depth / f_y
    z_cam = corrected_depth

    points_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1)

    ones = torch.ones((H, W, 1), device=device, dtype=points_cam.dtype)
    points_cam_hom = torch.cat([points_cam, ones], dim=-1)

    points_cam_hom = points_cam_hom.view(-1, 4)

    # Delete invalid points
    valid_mask = (depth.view(-1) > 0)
    points_cam_hom = points_cam_hom[valid_mask]

    v2w = camera.world_view_transform.transpose(0,1)
    cam2world = torch.inverse(v2w.to(device))
    points_world_hom = (cam2world @ points_cam_hom.t()).t()  # [N, 4]
    points_world = points_world_hom[:, :3]

    return points_world

def normal_filter(base_position, env_points, env_normal, gravity, normal_threshold = 0.7):
    base_position_clone = base_position.clone().detach().cpu().numpy()
    env_points_clone = env_points.clone().detach().cpu().numpy()
    env_normal_clone = env_normal.clone().detach().cpu().numpy()
    res_position = np.array([[-1000.0, -1000.0, -1000.0]])
    ori_num = base_position_clone.shape[0]

    count=0

    neighbors = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=1).fit(env_points_clone)
    dist, indices = neighbors.kneighbors(base_position_clone)
    print("Filtering points...")
    # for idx, n_idx in enumerate(indices):
    for idx, n_idx in enumerate(tqdm(indices, desc="filtering points...")):
        env_n = env_normal_clone[n_idx] # [3,]
        if np.dot(gravity, env_n[0]) <= normal_threshold or dist[idx][0] > 0.03:
            base_position_clone[idx][2]=10000.0
            count+=1

    mask = base_position_clone[:,2]<9999.0
    base_position_clone = base_position_clone[mask]

    res_position = torch.from_numpy(base_position_clone).float().to("cuda")
    res_position = res_position.view(-1,3)
    print("number of snow positions: ", ori_num-count)
    return res_position


def combine_sub_snow_anisotropic(base_position, scale, rots, opacities, g_vec, k=10, max_dist=0.3, method="kn", limit=500000):
    base_position_clone = base_position.clone().detach().cpu().numpy()
    scale_clone = scale.clone().detach().cpu().numpy()
    rots_clone = rots.clone().detach().cpu().numpy()
    opacities_clone = opacities.clone().detach().cpu().numpy()
    opacities_clone_add = opacities_clone
    res_position = np.array([[-10.0, -10.0, -10.0]])
    res_scale = np.array([[-10.0, -10.0, -10.0]])
    res_rots = np.array([[1.0, 0.0, 0.0, 0.0]])
    res_opacities = np.array([[-10.0]])
    max_threshold_list = np.linspace(0.3, 1.0, num=k)
    max_threshold_list = max_threshold_list * max_dist
    limit -= base_position_clone.shape[0]
    print("before combination: ", base_position_clone.shape[0])

    assert method in ["kn", "kd"]
    rotation_angles = []
    rotation_axes = []
    general_axe = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    
    if method == "kn":
        neighbors = NearestNeighbors(n_neighbors=k+1, radius=max_threshold_list[-1], algorithm='auto', n_jobs=1).fit(base_position_clone)
        distances, indices = neighbors.kneighbors(base_position_clone)
        for n in range(1,k+1):
            # print("iter",n)
            max_threshold = max_threshold_list[n-1]
            combined_index = np.zeros(base_position_clone.shape[0])
            for idx, n_idx in enumerate(tqdm(indices, desc="combining snow balls for iter "+str(n))):
                n_idx = n_idx[n]
                if combined_index[idx] and combined_index[n_idx]:
                    continue

                center = base_position_clone[idx]
                neighbour = base_position_clone[n_idx]
                n_vec = neighbour - center
                dist = np.linalg.norm(n_vec)
                direction_vec = n_vec / dist
                if dist > max_threshold:
                    continue
                if np.abs(np.dot(direction_vec, g_vec)) > 0.4:
                    continue
                s_1_log = scale_clone[idx]
                new_opacity = opacities_clone[idx]

                s_1 = np.exp(s_1_log)
                
                new_scale_length = s_1*np.max([np.min([dist/4.0/s_1[0], 2.0]), 1.0])
                new_scale_length[2] = dist/2.5
                new_scale = np.log(new_scale_length)

                new_position = (center + neighbour) / 2.0

                angle_between = np.arccos(np.clip(np.dot(n_vec / dist, z_axis), -1.0, 1.0))
                rotation_axis = np.cross(n_vec / dist, z_axis)
                if np.linalg.norm(rotation_axis) > 0:
                    rotation_axis /= np.linalg.norm(rotation_axis, keepdims=True)
                    rotation_angles.append(angle_between)
                    rotation_axes.append(rotation_axis)
                else:
                    rotation_angles.append(0)
                    rotation_axes.append(np.array([1, 0, 0]))

                res_position = np.concatenate((res_position, [new_position]), axis=0)
                res_scale = np.concatenate((res_scale, [new_scale]), axis=0)
                res_opacities = np.concatenate((res_opacities, [new_opacity]), axis=0)
                
                
                if res_position.shape[0] > limit:
                    break
            if res_position.shape[0] > limit:
                    break
    else:
        tree = KDTree(base_position_clone)
        for n in range(k):
            print("iter",n+1)
            max_threshold = max_threshold_list[n]
            # min_threshold = 0.0 if not n else max_threshold_list[n-1]
            min_threshold = max_threshold/2
            for idx in tqdm(range(base_position_clone.shape[0]), desc="kdtree processing..."):
                center = base_position_clone[idx]
                indices = tree.query_radius([center], r=max_threshold)
                neighbours = base_position_clone[indices[0]]
                distances = np.linalg.norm(neighbours - center, axis=1)
                dist_mask = (distances >= min_threshold)
                filtered_points = neighbours[dist_mask]
                o_list = opacities_clone[indices[0]]
                o_list = o_list[dist_mask]

                if filtered_points.shape[0]==0:
                    continue
                neighbour = filtered_points[0]
                s_1_log = scale_clone[idx]
                o_1 = np_sigmoid(opacities_clone[idx])
                o_2 = np_sigmoid(o_list[0])
                new_opacity = (o_1+o_2) / 1.5
                new_opacity = np_inv_sigmoid(new_opacity)
                s_1 = np.exp(s_1_log)

                n_vec = neighbour - center
                dist = np.linalg.norm(n_vec)
                if dist > max_threshold:
                    continue
                direction_vec = n_vec / dist
                new_scale_length = s_1 * 2.0
                new_scale_length[2] = dist/2.0
                new_scale = np.log(new_scale_length)
                new_position = (center + neighbour) / 2.0
                z_axis = np.array([0, 0, 1])
                angle_between = np.arccos(np.clip(np.dot(n_vec / dist, z_axis), -1.0, 1.0))
                rotation_axis = np.cross(n_vec / dist, z_axis)
                if np.linalg.norm(rotation_axis) > 0:
                    rotation_axis /= np.linalg.norm(rotation_axis, keepdims=True)
                    rotation_angles = np.concatenate((rotation_angles, [angle_between]),axis=0)
                    rotation_axes = np.concatenate((rotation_axes, [rotation_axis]),axis=0)
                else:
                    rotation_angles = np.concatenate((rotation_angles, [0]),axis=0)
                    rotation_axes = np.concatenate((rotation_axes, [general_axe]),axis=0)

                res_position = np.concatenate((res_position, [new_position]), axis=0)
                res_scale = np.concatenate((res_scale, [new_scale]), axis=0)
                res_opacities = np.concatenate((res_opacities, [new_opacity]), axis=0)

                if res_position.shape[0] > limit:
                    break
            if res_position.shape[0] > limit:
                    break

    print("after combination: ", res_position.shape[0] + base_position_clone.shape[0])
    print("Doing rotation operation...")

    angles = np.array(rotation_angles)
    axes = np.array(rotation_axes)
    
    half_angles = angles / 2
    rotation_quaternions = np.column_stack((
        np.cos(half_angles),
        axes[:, 0] * np.sin(half_angles),
        axes[:, 1] * np.sin(half_angles),
        axes[:, 2] * np.sin(half_angles)
    ))

    rots_base = np.zeros(rotation_quaternions.shape)
    rots_base[:, 0] = 1.0
    new_rots = quaternion_multiply_batch(rots_base, rotation_quaternions)
    res_rots = np.append(res_rots, new_rots, axis=0)

    res_position = np.append(res_position, base_position_clone, axis=0)
    res_scale = np.append(res_scale, scale_clone, axis=0)
    res_rots = np.append(res_rots, rots_clone, axis=0)
    res_opacities = np.append(res_opacities, opacities_clone_add, axis=0)

    res_position = torch.from_numpy(res_position).float().to("cuda")
    res_scale = torch.from_numpy(res_scale).float().to("cuda")
    res_rots = torch.from_numpy(res_rots).float().to("cuda")
    res_opacities = torch.from_numpy(res_opacities).float().to("cuda")
    return res_position, res_scale, res_rots, res_opacities

def quaternion_multiply_batch(q1, q2):
    """Multiply two batches of quaternions."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ]).T
