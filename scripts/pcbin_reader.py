import struct
import os

def read_point3d_binary(filepath):
    """
    Reads COLMAP's point3D.bin file and returns a dictionary with point data.
    """
    with open(filepath, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        points = {}
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<fff", f.read(12))  # x, y, z
            rgb = struct.unpack("<BBB", f.read(3))  # r, g, b
            error = struct.unpack("<d", f.read(8))[0]  # reprojection error
            track_length = struct.unpack("<Q", f.read(8))[0]
            track = [struct.unpack("<ii", f.read(8)) for _ in range(track_length)]  # image_id, point2D_idx
            points[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
        return points

def save_as_ply(points, output_file):
    """
    Saves the points dictionary as a PLY file.
    """
    with open(output_file, "w") as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Write points
        for point in points.values():
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

# Input and output paths
input_file = "/mnt/pool/sc/datasets/3d_recons/garden_pin/sparse/0/points3D.bin"
output_file = os.path.join(os.path.dirname(input_file), "points3D.ply")

# Read and convert
points = read_point3d_binary(input_file)
save_as_ply(points, output_file)

print(f"Saved PLY file to {output_file}")
