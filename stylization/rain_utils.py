import torch
from torch import nn
import numpy as np
from scene.gaussian_model import GaussianModel
from utils.sh_utils import RGB2SH
from stylization.rotation_utils import init_flake_position, apply_rotation, quaternion_to_rotation_matrix_batch, rotate_quaternions_to_gravity

def random_size_offset_rain(flake_number, factor=0.3):
    return (torch.rand([flake_number, 3], device="cuda") - 0.5) * factor

def init_rain_gaussians(droplets, sub_division, sky_color=None, radi=500, random_size=False, gravity = None):
    rain_list = []
    num_list = []
    # rot_list = np.array([]).reshape([])
    times = int(droplets // sub_division)
    total_num_flake = 0
    print("\nStart initiating rain gaussians.")
    if times:
        for i in range(times + 1):
            rain_gc = RainGaussian(3, None)
            if i == times:
                flake_number, rot_matrix = rain_gc.create_rain(droplet_num=droplets % sub_division, radi=radi, sky_color=sky_color, gravity=gravity)
                expanded_matrices = np.zeros((sub_division, 3, 3))
                expanded_matrices[:flake_number] = rot_matrix
                rot_matrix = expanded_matrices
            else:
                flake_number, rot_matrix = rain_gc.create_rain(droplet_num=sub_division, radi=radi, sky_color=sky_color, gravity=gravity)
            if random_size:
                size_range = random_size_offset_rain(flake_number)
                rain_gc._scaling += size_range
            rain_list.append(rain_gc)
            num_list.append(flake_number)
            
            # initiate or conactenate matrix
            if not i:
                rot_list = np.array([rot_matrix])
            else:
                rot_list = np.append(rot_list, [rot_matrix], axis=0)
            total_num_flake+=flake_number
    else:
        rain_gc = RainGaussian(3, None)
        flake_number, rot_matrix = rain_gc.create_rain(droplet_num=droplets, radi=radi, sky_color=sky_color, gravity=gravity)
        if random_size:
            size_range = random_size_offset_rain(flake_number)
            rain_gc._scaling += size_range
        rain_gc._scaling += size_range
        rain_list.append(rain_gc)
        num_list.append(flake_number)
        rot_list = np.array([rot_matrix])
        total_num_flake+=flake_number
    print(total_num_flake, "droplets generated.")
    return rain_list, num_list, rot_list

def gen_rainfall_vector(rotation_list, flake_number, idx, fall_speed = 3.5, gravity=None):
    rain_fall = np.zeros([flake_number, 3])
    rain_fall[:, 1] = fall_speed
    # print(rot_list[i].shape, rain_fall.shape)
    rain_fall = np.einsum('nij,nj->ni', rotation_list[idx, :flake_number], rain_fall)
    if gravity is not None:
        default_gravity = np.array([0, -1, 0], dtype=np.float32)
        gravity = gravity / np.linalg.norm(gravity)
        
        axis = np.cross(default_gravity, gravity)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(default_gravity, gravity), -1.0, 1.0))  # 夹角
            
            # Generate rotation matrix
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
            
            # Apply rotation to rain_fall
            rain_fall = np.dot(rain_fall, rotation_matrix.T)
    rain_fall = torch.from_numpy(rain_fall).float().cuda()
    return rain_fall

class RainGaussian(GaussianModel):
    def __init__(self, sh_degree : int, args):
        super().__init__(sh_degree, args)
    
    def create_rain(self, droplet_num=10000, radi=300, sky_color=None, gravity=None):
        num_flakes = droplet_num
        self.spatial_lr_scale = 0

        if self._xyz.shape[0] > 0:
            tmp_xyz = (self._xyz).cpu().numpy()
            center = np.mean(tmp_xyz)
            radii = np.linalg.norm(tmp_xyz - center, axis=1)
            median_radius = np.median(radii)
            max_radius = np.max(radii)
            mean_radius = np.mean(radii)
            print(median_radius, max_radius)
            rain_position = init_flake_position(num_flakes, mean_radius * 2)
        else:
            rain_position = init_flake_position(num_flakes, radi)
        rain_point_cloud = torch.tensor(rain_position).float().cuda()
        if sky_color is None:
            rain_color = torch.tensor([100,100,100]).repeat(rain_point_cloud.shape[0], 1) / 255.0
        else:
            rain_color = sky_color.repeat(rain_point_cloud.shape[0], 1)
        rain_color = RGB2SH(rain_color.float().cuda())
        features = torch.zeros((rain_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = rain_color
        features[:, 3:, 1:] = 0.0
        scales = np.ones(rain_point_cloud.shape) * (-4.5)
        scales = torch.tensor(scales).float().cuda()
        scales[:, 1] *= 0.4
        # rots = torch.zeros((snow_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        np_rots = np.zeros((rain_point_cloud.shape[0], 4))
        # np_rots[:, 0] = 1.0
        np_rots[:, 0] = np.cos(np.pi/32)
        np_rots[:, 3] = np.sin(np.pi/32)
        np_rots = apply_rotation(np_rots, if_snow=False, max_xy_angle_degrees=5)
        if gravity is not None:
            np_rots = rotate_quaternions_to_gravity(np_rots, gravity)
        rots = torch.from_numpy(np_rots).float().cuda()
        # print(torch.mean(self._opacity), torch.max(self._opacity))
        mean_opacity = torch.mean(self._opacity)
        # opacities = inverse_sigmoid(1 * torch.ones((snow_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = torch.ones((rain_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * (-1.0)
        normals = torch.ones((rain_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        flake_xyz = nn.Parameter(rain_point_cloud.requires_grad_(False))
        flake_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False))
        flake_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))
        flake_scaling = nn.Parameter(scales.requires_grad_(False))
        flake_rotation = nn.Parameter(rots.requires_grad_(False))
        flake_opacity = nn.Parameter(opacities.requires_grad_(False))
        flake_normal = nn.Parameter(normals.requires_grad_(False))
        flakemax_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        flake_deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
        self._xyz = torch.cat((self._xyz.to("cuda"), flake_xyz))
        self._features_dc = torch.cat((self._features_dc.to("cuda"), flake_features_dc))
        self._features_rest = torch.cat((self._features_rest.to("cuda"), flake_features_rest))
        self._scaling = torch.cat((self._scaling.to("cuda"), flake_scaling))
        self._rotation = torch.cat((self._rotation.to("cuda"), flake_rotation))
        self._opacity = torch.cat((self._opacity.to("cuda"), flake_opacity))
        self._normal = torch.cat((self._normal.to("cuda"), flake_normal))
        self.max_radii2D = torch.cat((self.max_radii2D.to("cuda"), flakemax_radii2D))
        self._deformation_table = torch.cat((self._deformation_table.to("cuda"), flake_deformation_table))

        rot_matrix = quaternion_to_rotation_matrix_batch(np_rots)
        # return torch.from_numpy(rot_matrix).float().cuda()
        return rain_point_cloud.shape[0], rot_matrix