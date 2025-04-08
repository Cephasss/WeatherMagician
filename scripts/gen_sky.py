import open3d as o3d
import sys
import numpy as np

def add_sky_points(num_points, radius, center, color, normal):
    phi = np.random.uniform(0, np.pi, num_points)  # 角度 phi
    theta = np.random.uniform(np.pi, 2* np.pi, num_points)  # 角度 theta
    theta = np.random.uniform(0, 2* np.pi, num_points)  # 360

    # 球坐标转笛卡尔坐标
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.vstack((x, y, z)).T
    points += np.array(center)
    colors = np.tile(color, (num_points, 1))
    normals = np.tile(normal, (num_points, 1))
    return points, colors, normals
def process_ply_file(input_file, output_file, num_points):

    # 读取输入的ply文件
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Total points: {len(pcd.points)}")
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)
    original_normals = np.asarray(pcd.normals)

    # 计算场景的最大半径
    center = np.mean(original_points, axis=0)
    radii = np.linalg.norm(original_points - center, axis=1)
    median_radius = np.median(radii)
    max_radius = np.max(np.linalg.norm(original_points - center, axis=1))
    max_radius = 15 * max_radius
    median_radius = 15 *median_radius
    hemisphere_color = [0.5, 0.5, 0.5]  # 半球点的颜色 (灰色)
    hemisphere_normal = [0, 0, -1]  # 半球点的法线方向

    points, colors, normals = add_sky_points(num_points, median_radius, center, hemisphere_color, hemisphere_normal)

    combined_points = np.vstack((original_points, points))
    combined_colors = np.vstack((original_colors, colors))
    combined_normals = np.vstack((original_normals, normals))

    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    combined_pcd.normals = o3d.utility.Vector3dVector(combined_normals)
    print(f"After adding sky, total points: {len(combined_pcd.points)}")

    o3d.io.write_point_cloud(output_file, combined_pcd)

# 使用函数
process_ply_file(sys.argv[1], sys.argv[2], 30000)