from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Sobel definition
sobel_x = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

sobel_y = torch.tensor([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

# kernel definition
gaussian_kernel = torch.tensor([[1/16, 2/16, 1/16],
                                [2/16, 4/16, 2/16],
                                [1/16, 2/16, 1/16]], dtype=torch.float32).view(1, 1, 3, 3)
gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

def compute_normal_map(depth_map):
    """
    :param depth_map: [1, w, h]
    :return normal: [3, w, h]
    """
    depth_map = depth_map.unsqueeze(0)
    sobel_x_device = sobel_x.to(depth_map.device)
    sobel_y_device = sobel_y.to(depth_map.device)
    gaussian_kernel_device = gaussian_kernel.to(depth_map.device)

    dx = F.conv2d(depth_map, sobel_x_device, padding=1)
    dy = F.conv2d(depth_map, sobel_y_device, padding=1)
    normal_x = torch.zeros([3,dx.shape[2],dx.shape[3]]).to(depth_map.device)
    normal_y = torch.zeros([3,dx.shape[2],dx.shape[3]]).to(depth_map.device)
    normal_x[0, :, :] = dx[0]
    normal_y[1, :, :] = dy[0]
    dz = torch.cross(normal_x.permute(1,2,0), normal_y.permute(1,2,0),dim=2)

    normal_x = dx
    normal_y = -dy
    
    normal_z = dz.permute(2,0,1)[2].unsqueeze(0)
    epsilon = 1e-8

    norm = torch.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2 + epsilon)
    normal_map = torch.cat((normal_x / norm, normal_y / norm, normal_z / norm), dim=1)  # [1, 3, w, h]

    normal_map_smoothed = F.conv2d(normal_map, gaussian_kernel_device, padding=1, groups=3)
    final_normal_map = normal_map_smoothed.squeeze(0)

    threshold = 1e-2

    near_zero_mask = (final_normal_map.abs() < threshold).all(dim=0)
    value_to_replace = torch.tensor([1.0, 0.0, 0.0]).to(final_normal_map.device).view(3, 1)  # [3, 1]
    final_normal_map[:, near_zero_mask] = value_to_replace.expand(-1, near_zero_mask.sum())

    norm = torch.linalg.norm(final_normal_map, dim=0)
    final_normal_map/=norm




    return final_normal_map

def estimate_normals(points_t, k=5):
    points = points_t.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    normals = np.zeros(points.shape)

    for i, point in enumerate(tqdm(points, desc="Processing normal estimation")):
        indices = nbrs.kneighbors(point.reshape(1, -1), return_distance=False)[0]
        neighbors = points[indices]
        
        cov_matrix = np.cov(neighbors - point, rowvar=False)
        
        _, _, vh = np.linalg.svd(cov_matrix)
        normals[i] = vh[2]
        
    return torch.from_numpy(normals).float().to("cuda")

def estimate_normals_pca(points, k=100):
    """
    :param k: knn numbers
    :return: normals [n,3]
    """
    points = points.detach().cpu().numpy()
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1).fit(points)
    distances, indices = neighbors.kneighbors(points)

    normals = np.zeros_like(points)

    for i, neighbors_idx in enumerate(indices):
        neighbor_points = points[neighbors_idx]

        pca = PCA(n_components=3)
        pca.fit(neighbor_points)

        normal = pca.components_[-1]
        normals[i] = normal

    return torch.from_numpy(normals)

def visualize_normal_map(normal_map):
    normal_map_vis = normal_map.permute(1, 2, 0)
    normal_map_vis = (normal_map_vis + 1) / 2
    
    plt.imshow(normal_map_vis.detach().cpu().numpy())
    plt.title('Normal Map')
    plt.show()