#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
from stylization.rotation_utils import *
from stylization.snow_utils import filter_outlier, estimate_gravity_direction, init_scale
from stylization.snow_utils import local_plane_densification, compute_average_distance, depth_to_point_cloud, normal_filter
from stylization.rotation_utils import init_flake_position, apply_rotation


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def normalize_normals(normals):
            """
            Normalize the normals
            :param normals: [n, 3]
            :return: normalized normals, [n, 3]
            """
            norms = torch.linalg.norm(normals, dim=1)  # [n, 1]
            normalized_normals = normals / (norms.view(-1,1) + 1e-8) 

            return normalized_normals
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.normal_normalization = normalize_normals
        self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        if args is not None:
            self._deformation = deform_network(args)
        else:
            self._deformation = None
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._deformation_table,
        
        # self.grid,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._normal,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_normal(self):
        return self.normal_normalization(self._normal)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert a quaternion into a rotation matrix.
        Quaternion format: [r, x, y, z]
        """
        r, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

        R = torch.zeros((quaternion.shape[0], 3, 3), device=quaternion.device)
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R

    def compute_normals(self, scale, rot, means, campos):
        """
        Compute normals for a set of points.

        Args:
            scale (torch.Tensor): Scaling coefficients of shape [n, 3].
            rot (torch.Tensor): Rotation quaternions of shape [n, 4].
            means (torch.Tensor): Point positions of shape [n, 3].
            campos (torch.Tensor): Camera position of shape [3].

        Returns:
            normals (torch.Tensor): Normal vectors of shape [n, 3].
        """
        n = scale.shape[0]

        # Convert quaternions to rotation matrices
        R = self.quaternion_to_rotation_matrix(rot)

        # Determine initial normals based on scale
        initial_normals = torch.zeros((n, 3), device=scale.device)
        condition1 = (scale[:, 0] > scale[:, 2]) & (scale[:, 1] > scale[:, 2])
        condition2 = (scale[:, 0] > scale[:, 1]) & (scale[:, 2] > scale[:, 1])

        initial_normals[condition1] = torch.tensor([0.0, 0.0, 1.0], device=scale.device)
        initial_normals[condition2] = torch.tensor([0.0, 1.0, 0.0], device=scale.device)
        initial_normals[~(condition1 | condition2)] = torch.tensor([1.0, 0.0, 0.0], device=scale.device)

        # Rotate normals
        rotated_normals = torch.einsum('nij,nj->ni', R, initial_normals)
        # Compute ray directions from camera to points
        ray_directions = means - campos

        # Ensure normals face the camera
        dot_products = torch.einsum('ni,ni->n', ray_directions, rotated_normals)
        flipped_normals = torch.where(dot_products.unsqueeze(1) > 0, -rotated_normals, rotated_normals)
        return flipped_normals

    def create_acc_snow(self, threshold=0.5, big_scale = False):
        assert self._normal.shape[0] == self.xyz.shape[0]

        # init snow locations based on normal dot operation
        xyz = self.get_xyz
        scaling = self.get_scaling
        rotation = self.get_rotation
        campos = torch.tensor([0.0, -25.0, 0.0], device='cuda')
        normal = (self.compute_normals(scaling, rotation, xyz,campos))
        R = self.quaternion_to_rotation_matrix(rotation)
        gravity_vec = estimate_gravity_direction(xyz)
        # gravity_vec = np.array([0.0,-1.0,-0.0])
        print("negative gravity estimated: ", gravity_vec)
        distance = torch.norm(xyz, dim=1)
        dot_vec = torch.from_numpy(gravity_vec).float().to("cuda")
        n = normal.unsqueeze(1)
        d = dot_vec.unsqueeze(1)
        normal_dot = (torch.matmul(n, d))
        init_mask = (normal_dot > threshold).squeeze()

        init_snow_position = torch.zeros([(self._xyz[init_mask]).shape[0], 3]).float().cuda()
        if init_snow_position.shape[0]==0:
            print("Initialization failed: No ground base found.")
            return -1
        init_snow_position += self._xyz[init_mask]
        if big_scale:
            init_snow_position += normal[init_mask] / 100.0
        else:
            init_snow_position += normal[init_mask] / 300.0
            # init_snow_position += torch.from_numpy(gravity_vec).float().cuda() / 200.0
        
        distance = torch.norm(init_snow_position, dim=1)
        snow_position = init_snow_position
        
        sample_size = 200000
        if snow_position.shape[0]>sample_size:
        
            indices = torch.randperm(snow_position.size(0))[:sample_size]
            sub_snow = snow_position[indices]
        average_dist,_ = compute_average_distance(snow_position)
        print(average_dist)

        # Plane densification
        if big_scale:
            snow_position = local_plane_densification(snow_position, radius=0.4, g_vec=gravity_vec, samples_per_plane=50, iteration=1)
        else:
            # for smaller scale
            snow_position = local_plane_densification(snow_position, radius=average_dist/2, g_vec=gravity_vec, samples_per_plane=15, iteration=1, algorithm='auto', noise_intensity=0.0005)

        print("Final snow particles number: ",snow_position.shape[0])
        
        if big_scale:
            scales = init_scale(snow_position, xyz, self._scaling, base_scale=-4.5)
        else:
            scales = init_scale(snow_position, xyz, self._scaling, base_scale=-6.0) # last 5.2->5.6
        opacities = torch.ones((snow_position.shape[0], 1), dtype=torch.float, device="cuda") * (0.75)
        rots = torch.zeros((snow_position.shape[0], 4), dtype=torch.float).to("cuda")
        rots[:, 0] = 1.0
        # _, inlier_mask = filter_outlier(snow_position, k=3, threshold=0.02)
        # snow_position = snow_position[inlier_mask]
        # opacities = opacities[inlier_mask]
        # scales = scales[inlier_mask] 
        # rots = rots[inlier_mask]
        
        _, inlier_mask = filter_outlier(snow_position, k=15, threshold=average_dist/1.5)
        snow_position = snow_position[inlier_mask]
        opacities = opacities[inlier_mask]
        scales = scales[inlier_mask].to("cuda")
        rots = rots[inlier_mask].to("cuda")
        

        snow_color = torch.tensor([230, 230, 230]).repeat(snow_position.shape[0], 1) / 255.0
        # snow_color = torch.tensor([130, 120, 80]).repeat(snow_position.shape[0], 1) /255.0
        snow_color = RGB2SH(snow_color.float().cuda())
        features = torch.zeros((snow_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = snow_color
        features[:, 3:, 1:] = 0.4/0.28


        normals = torch.zeros(snow_position.shape).float().cuda()
        normals += torch.from_numpy(gravity_vec).cuda()
        # normals = normal.to("cuda")

        snow_xyz = nn.Parameter(snow_position.requires_grad_(False))
        # self.grid = self.grid.to("cuda")
        snow_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False))
        snow_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))
        snow_scaling = nn.Parameter(scales.requires_grad_(False))
        snow_normal = nn.Parameter(normals.requires_grad_(False))
        snow_rotation = nn.Parameter(rots.requires_grad_(False))
        snow_opacity = nn.Parameter(opacities.requires_grad_(False))
        snowmax_radii2D = torch.zeros((snow_position.shape[0]), device="cuda")
        snow_deformation_table = torch.gt(torch.ones((snow_position.shape[0]), device="cuda"), 0)
        # self.densification_postfix(flake_xyz,flake_features_dc,flake_features_rest,flake_opacity,flake_scaling,flake_rotation,flake_deformation_table)
        self._xyz = torch.cat((self._xyz.to("cuda"), snow_xyz.to("cuda")))
        self._features_dc = torch.cat((self._features_dc.to("cuda"), snow_features_dc))
        self._features_rest = torch.cat((self._features_rest.to("cuda"), snow_features_rest))
        self._scaling = torch.cat((self._scaling.to("cuda"), snow_scaling.to("cuda")))
        self._normal = torch.cat((self._normal.to("cuda"), snow_normal))
        self._rotation = torch.cat((self._rotation.to("cuda"), snow_rotation))
        self._opacity = torch.cat((self._opacity.to("cuda"), snow_opacity))
        self.max_radii2D = torch.cat((self.max_radii2D.to("cuda"), snowmax_radii2D))
        self._deformation_table = torch.cat((self._deformation_table.to("cuda"), snow_deformation_table))

    def create_acc_snow_from_depth(self, depth_list, view_list, threshold=0.5, big_scale = False):
        xyz = self.get_xyz
        scaling = self.get_scaling
        rotation = self.get_rotation
        campos = torch.tensor([0.0, -25.0, 0.0], device='cuda')
        normal = (self.compute_normals(scaling, rotation, xyz,campos))
        gravity_vec = estimate_gravity_direction(xyz)
        # gravity_vec = np.array([-0.0, -1.0, -0.0],dtype=np.float32)
        print("negative gravity estimated: ", gravity_vec)
        distance = torch.norm(xyz, dim=1)
        dot_vec = torch.from_numpy(gravity_vec).float().to("cuda")

        init_snow_position = depth_to_point_cloud(view_list[0],depth_list[0])
        for i in range(1, len(view_list)):
            init_snow_position = torch.cat([init_snow_position, depth_to_point_cloud(view_list[i],depth_list[i])], dim=0)
        init_snow_position = normal_filter(init_snow_position,xyz,normal,gravity_vec,threshold)
        # verify_depth_reprojection(view, depth, init_snow_position)
        # init_snow_position = torch.zeros([(self._xyz[init_mask]).shape[0], 3]).float().cuda()
        if init_snow_position.shape[0]==0:
            print("Initialization failed: No ground base found.")
            return -1

        
        init_snow_position += dot_vec/300.0
        distance = torch.norm(init_snow_position, dim=1)
        dist = torch.norm(xyz, dim=1)
        snow_position = init_snow_position

        sample_size = 200000
        if snow_position.shape[0]>sample_size:
        
            indices = torch.randperm(snow_position.size(0))[:sample_size]
            sub_snow = snow_position[indices]
        average_dist,_ = compute_average_distance(snow_position)
        print(average_dist)

        if big_scale:
            scales = init_scale(snow_position, xyz, self._scaling, base_scale=-4.5)
        else:
            scales = init_scale(snow_position, xyz, self._scaling, base_scale=-7.3)
        opacities = torch.ones((snow_position.shape[0], 1), dtype=torch.float, device="cuda") * (0.75)
        rots = torch.zeros((snow_position.shape[0], 4), dtype=torch.float).to("cuda")
        rots[:, 0] = 1.0
        _, inlier_mask = filter_outlier(snow_position, k=3, threshold=0.04)
        snow_position = snow_position[inlier_mask]
        opacities = opacities[inlier_mask]
        scales = scales[inlier_mask] 
        rots = rots[inlier_mask]
        
        # _, inlier_mask = filter_outlier(snow_position, k=10, threshold=average_dist/1.5)
        # snow_position = snow_position[inlier_mask]
        # opacities = opacities[inlier_mask]
        # scales = scales[inlier_mask].to("cuda")
        # rots = rots[inlier_mask].to("cuda")

        snow_color = torch.tensor([255, 255, 255]).repeat(snow_position.shape[0], 1) / 255.0
        # snow_color = torch.tensor([130, 120, 80]).repeat(snow_position.shape[0], 1) /255.0
        snow_color = RGB2SH(snow_color.float().cuda())
        features = torch.zeros((snow_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = snow_color
        # features[:, 3:, 1:] = 0.4/0.28

        normals = torch.zeros(snow_position.shape).float().cuda()
        normals += torch.from_numpy(gravity_vec).cuda()
        # normals = normal.to("cuda")

        snow_xyz = nn.Parameter(snow_position.requires_grad_(False))
        # self.grid = self.grid.to("cuda")
        snow_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(False))
        snow_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(False))
        snow_scaling = nn.Parameter(scales.requires_grad_(False))
        snow_normal = nn.Parameter(normals.requires_grad_(False))
        snow_rotation = nn.Parameter(rots.requires_grad_(False))
        snow_opacity = nn.Parameter(opacities.requires_grad_(False))
        snowmax_radii2D = torch.zeros((snow_position.shape[0]), device="cuda")
        snow_deformation_table = torch.gt(torch.ones((snow_position.shape[0]), device="cuda"), 0)
        # self.densification_postfix(flake_xyz,flake_features_dc,flake_features_rest,flake_opacity,flake_scaling,flake_rotation,flake_deformation_table)
        self._xyz = torch.cat((self._xyz.to("cuda"), snow_xyz.to("cuda")))
        self._features_dc = torch.cat((self._features_dc.to("cuda"), snow_features_dc))
        self._features_rest = torch.cat((self._features_rest.to("cuda"), snow_features_rest))
        self._scaling = torch.cat((self._scaling.to("cuda"), snow_scaling.to("cuda")))
        self._normal = torch.cat((self._normal.to("cuda"), snow_normal))
        self._rotation = torch.cat((self._rotation.to("cuda"), snow_rotation))
        self._opacity = torch.cat((self._opacity.to("cuda"), snow_opacity))
        self.max_radii2D = torch.cat((self.max_radii2D.to("cuda"), snowmax_radii2D))
        self._deformation_table = torch.cat((self._deformation_table.to("cuda"), snow_deformation_table))

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
                snow_color = torch.tensor([200,200,200]).repeat(snow_point_cloud.shape[0], 1) / 255.0
            else:
                snow_color = sky_color.repeat(snow_point_cloud.shape[0], 1)
            snow_point_cloud = torch.tensor(snow_point_cloud).float().cuda()
            snow_color = RGB2SH(snow_color.float().cuda())
            features = torch.zeros((snow_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = snow_color
            features[:, 3:, 1:] = 0.0
            scales = np.ones(snow_point_cloud.shape) * (-5.85)

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
            opacities = torch.ones((snow_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * (10.3)
            normals = torch.ones((snow_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
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


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # print(pcd.colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        normal = torch.ones((fused_point_cloud.shape[0], 3), device="cuda")
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._normal.shape[1]):
            l.append('normal_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # normal = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, normal, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normal_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("normal_")]
        normal_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        normal = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(normal_names):
            normal[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._normal = optimizable_tensors["normal"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_normal, new_rotation, new_deformation_table):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "normal" : new_normal,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._normal = optimizable_tensors["normal"]
        self._rotation = optimizable_tensors["rotation"]
        # self._deformation = optimizable_tensors["deformation"]
        
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_normal = self.get_normal[selected_pts_mask].repeat(N,1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_normal, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_normal, new_rotation, new_deformation_table)

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
    

        return final_point, mask_d    
    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)

        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_normal = self._normal[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_normal, new_rotation, new_deformation_table)
        return selected_xyz, new_xyz

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        normals = self._normal.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, normals, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
