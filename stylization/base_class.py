from scene.gaussian_model import GaussianModel
import torch
import numpy as np
from utils.sh_utils import RGB2SH
from torch import nn
from stylization.snow_utils import random_size_offset
from stylization.rotation_utils import init_flake_position, apply_rotation
from tqdm import tqdm


def init_snow_gaussians(fall_rate, sub_division, sky_color=None, radi=500, random_size=True):
    snow_list = []
    num_list = []
    divised_times = int(fall_rate//sub_division)
    total_num_flake = 0
    print("\nStart initiating snow gaussians.")
    if divised_times:
        for i in tqdm(range(divised_times+1),desc="generating sub gaussians"):
            snow_gc = SnowGaussian(3, None)
            if i == divised_times:
                flake_number = snow_gc.create_flake(fall_rate=fall_rate % sub_division, radi=radi, sky_color=sky_color)
            else:
                flake_number = snow_gc.create_flake(fall_rate=sub_division, radi=radi, sky_color=sky_color)
            if random_size:
                size_range = random_size_offset(flake_number)
                snow_gc._scaling += size_range
            snow_list.append(snow_gc)
            num_list.append(flake_number)
            total_num_flake+=flake_number
    else:
        snow_gc = SnowGaussian(3, None)
        flake_number = snow_gc.create_flake(fall_rate=fall_rate, radi=radi, sky_color=sky_color)
        if random_size:
            size_range = random_size_offset(flake_number)
            snow_gc._scaling += size_range
        snow_list.append(snow_gc)
        num_list.append(flake_number)
        total_num_flake+=flake_number
    print(total_num_flake, "snowflakes generated.")
    return snow_list, num_list

class SnowGaussian(GaussianModel):
    def __init__(self, sh_degree : int, args):
        super().__init__(sh_degree, args)
    
    def create_flake(self, fall_rate=10000, radi=300, dense=False, sky_color = None):
            if dense:
                mass_concentration = 0.30 * fall_rate
            else:
                mass_concentration = 0.47 * fall_rate
            num_flakes = int(mass_concentration / 0.2)
            # N0 = 2500 * fall_rate**(-0.94)
            # l = 22.9 * fall_rate**(-0.45)
            # ND = []
            # for D in range(1, 10):
            #     diameter = D/10.0
            #     ND.append(N0 * np.exp(-l*diameter))
            self.spatial_lr_scale = 0

            if self._xyz.shape[0] > 0:
                tmp_xyz = (self._xyz).cpu().numpy()
                center = np.mean(tmp_xyz)
                radii = np.linalg.norm(tmp_xyz - center, axis=1)
                median_radius = np.median(radii)
                max_radius = np.max(radii)
                mean_radius = np.mean(radii)
                print(median_radius, max_radius)
                snow_position = init_flake_position(num_flakes, mean_radius*2)
            else:
                snow_position = init_flake_position(num_flakes, radi)
            snow_point_cloud = torch.tensor(snow_position).float().cuda()
            snow_point_cloud = torch.cat([snow_point_cloud, snow_point_cloud, snow_point_cloud], dim=0)
            if sky_color is None:
                snow_color = torch.tensor([190,190,190]).repeat(snow_point_cloud.shape[0], 1) / 255.0
            else:
                snow_color = sky_color.repeat(snow_point_cloud.shape[0], 1)
            snow_point_cloud = torch.tensor(snow_point_cloud).float().cuda()
            # snow_color = torch.tensor([130, 120, 80]).repeat(snow_point_cloud.shape[0], 1) / 255.0
            snow_color = RGB2SH(snow_color.float().cuda())
            features = torch.zeros((snow_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = snow_color
            features[:, 3:, 1:] = 0.0
            # print("Number of snow flakes : ", snow_position.shape[0])

            # dist2 = torch.clamp_min(distCUDA2(snow_point_cloud.float().cuda()), 0.0000001)
            # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            scales = np.ones(snow_point_cloud.shape) * (-4.2)
            # scales = np.ones(snow_point_cloud.shape) * (-3.85)
            # scales = np.ones(snow_point_cloud.shape) * (-2.0)
            scales = torch.tensor(scales).float().cuda()
            if not dense:
                scales[:, 1] *= 0.76
                # scales[:, 1] *= 0.8
            # print(scales)
            # rots = torch.zeros((snow_point_cloud.shape[0], 4), device="cuda")
            # rots[:, 0] = 1
            rots = np.zeros((snow_point_cloud.shape[0], 4))
            rots[:, 0] = 1
            rots = torch.from_numpy(apply_rotation(rots, if_snow=True, max_xy_angle_degrees=180)).float().cuda()
            # print(torch.mean(self._opacity), torch.max(self._opacity))
            mean_opacity = torch.mean(self._opacity)
            # opacities = inverse_sigmoid(1 * torch.ones((snow_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities = torch.ones((snow_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * (1.3)
            normals = torch.ones((snow_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            flake_xyz = nn.Parameter(snow_point_cloud.requires_grad_(False))
            # self.grid = self.grid.to("cuda")
            flake_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False))
            flake_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))
            flake_scaling = nn.Parameter(scales.requires_grad_(False))
            flake_rotation = nn.Parameter(rots.requires_grad_(False))
            flake_opacity = nn.Parameter(opacities.requires_grad_(False))
            flake_normal = nn.Parameter(normals.requires_grad_(False))
            flakemax_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            flake_deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
            # self.densification_postfix(flake_xyz,flake_features_dc,flake_features_rest,flake_opacity,flake_scaling,flake_rotation,flake_deformation_table)
            self._xyz = torch.cat((self._xyz.to("cuda"), flake_xyz))
            self._features_dc = torch.cat((self._features_dc.to("cuda"), flake_features_dc))
            self._features_rest = torch.cat((self._features_rest.to("cuda"), flake_features_rest))
            self._scaling = torch.cat((self._scaling.to("cuda"), flake_scaling))
            self._rotation = torch.cat((self._rotation.to("cuda"), flake_rotation))
            self._opacity = torch.cat((self._opacity.to("cuda"), flake_opacity))
            self._normal = torch.cat((self._normal.to("cuda"), flake_normal))
            self.max_radii2D = torch.cat((self.max_radii2D.to("cuda"), flakemax_radii2D))
            self._deformation_table = torch.cat((self._deformation_table.to("cuda"), flake_deformation_table))

            return int(snow_position.shape[0])