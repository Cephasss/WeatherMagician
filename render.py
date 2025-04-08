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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time, sleep
from scipy.spatial.transform import Rotation as R, Slerp
from scene.cameras import Camera
import threading
import concurrent.futures
import copy
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import random
from utils.normal_utils import visualize_normal_map, compute_normal_map, estimate_normals
from stylization import *


def interpolate_views(ori_views, inter_num):
    views_list = [v for v in ori_views]
    new_views = [views_list[0]]
    for i in range(len(views_list) - 1):
        v1, v2 = views_list[i], views_list[i + 1]
        R1 = v1.R
        R2 = v2.R
        T1 = v1.T
        T2 = v2.T
        interval = 1 / (inter_num + 1)
        for j in range(inter_num):
            t = interval * (j + 1)
            T_j = (1 - t) * T1 + t * T2
            q1 = R.from_matrix(R1).as_quat()
            q2 = R.from_matrix(R2).as_quat()

            key_times = [0, 1]
            key_rots = R.from_quat([q1, q2])
            slerp = Slerp(key_times, key_rots)
            R_j = slerp([t]).as_matrix()[0]
            v_j = Camera(colmap_id=v1.colmap_id, R=R_j, T=T_j, FoVx=v1.FoVx, FoVy=v1.FoVy, image=v1.original_image,
                         gt_alpha_mask=None,
                         image_name=v1.image_name, uid=v1.uid, data_device=torch.device("cuda"),
                         time=(1 - t) * v1.time + t * v2.time,
                         mask=None, depth=None)
            new_views.append(v_j)
        new_views.append(v2)
    print("Frames after interpolation: ", len(new_views))
    return new_views

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            cv2.imwrite(os.path.join(path, '{0:05d}'.format(count) + ".png"), image * 255)
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, source_path, interpolate=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    style_path = os.path.join(model_path, name, "ours_{}".format(iteration), "style")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(style_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    render_images = []
    render_depths = []
    render_normals = []
    gt_list = []
    render_list = []
    depth_list = []
    style_list = []
    render_styles = []
    normal_list = []
    print("point nums:", gaussians._xyz.shape[0])


    bg_color = [0, 0, 0]
    background_render = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    init = 0
    # gaussians._normal = estimate_normals(gaussians._xyz, k=100).to("cuda")

    if args.acc_snow:
        save=0
        if args.acc_depth:
            # init snow from depth map
            from scene.cameras import Camera
            def get_rotation_matrix_from_gravity(gravity_vector):
                camera_initial_direction = torch.tensor([0.0, 0.0, 1.0])
                axis = torch.cross(camera_initial_direction, gravity_vector)
                if torch.norm(axis) < 1e-6:
                    return torch.eye(3)
                axis = axis / torch.norm(axis)
                cos_angle = torch.dot(camera_initial_direction, gravity_vector) / (
                    torch.norm(camera_initial_direction) * torch.norm(gravity_vector))
                angle = torch.acos(cos_angle)

                sin_angle = torch.sin(angle)
                cos_angle = torch.cos(angle)

                K = torch.tensor([[0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0]])

                R = torch.eye(3) * cos_angle + (1 - cos_angle) * torch.outer(axis, axis) + sin_angle * K
                return R
            g = torch.tensor([-0.0, -1.0, -0.0],dtype=torch.float32)
            R_matrix = get_rotation_matrix_from_gravity(-g)
            centroid = torch.mean(gaussians._xyz, dim=0).to("cpu")
            T_matrix = torch.tensor([-3.0,0.0,8.4])+40*g
            R_matrix=R_matrix.numpy()
            T_matrix=T_matrix.numpy()
            indice_number = len(views.dataset)
            indices = np.random.choice(indice_number, indice_number, replace=False)
            acc_views = [views[i] for i in indices]
            acc_depth_list=[]
            for m in range(len(acc_views)):
                ref_render_pkg = render(acc_views[m], gaussians, pipeline, background_render)
                acc_depth_list.append(ref_render_pkg["depth"])
            
            
            print(len(acc_views))
            gaussians.create_acc_snow_from_depth(acc_depth_list, acc_views, 0.5)
        else:
            gaussians.create_acc_snow(0.6)
            print("\n[ITER {}] Saving Snow cover point cloud".format(iteration+1))

        if save:
            ply_path = os.path.join(source_path, "point_cloud/iteration_{}".format(iteration)+"_snow")
            print(ply_path)
            gaussians.save_ply(os.path.join(ply_path, "point_cloud.ply"))
            gaussians.save_deformation(ply_path)
            return 0
    weights = torch.tensor([0.3333, 0.3333, 0.3333]).view(1, 3, 1, 1)
    weights = weights.to("cuda")
    total_g_num = gaussians._xyz.shape[0]
    bg_num = total_g_num//3.2
    acc_change=0
    if interpolate:
        views = interpolate_views(views, interpolate)
    max_frame = 100000
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 1 and init: time1 = time()
        if idx >= max_frame:
            continue

        render_pkg = render(view, gaussians, pipeline, background_render, cam_type=cam_type)
        rendering = render_pkg["render"]

        normal_map = render_pkg["normal"]
        visibility_filter = render_pkg["visibility_filter"]
        abs_rendering_depth = render_pkg["depth"]
        rendering_depth = torch.clamp(abs_rendering_depth / 100.0, 0.0, 1.0)
        # print(rendering_depth)

        if not init:
            init = 1
            sky_mask = rendering_depth > 0.9
            sky_mask = sky_mask.squeeze(0)
            p_rendering = rendering.permute(1,2,0)
            sky_color = torch.mean(p_rendering[sky_mask], dim=0)
            if sky_color[0].isnan():
                sky_color = torch.tensor([180, 180, 180],dtype=torch.float32)/255.0
            sky_color = sky_color.to("cuda")
            sky_brightness = torch.mean(sky_color)
            # sky_brightness = torch.mean(torch.tensor([230, 230, 230]).to("cuda")/255.0)
            print("\nsky color:", sky_color)
            gravity_vec = (-1) * estimate_gravity_direction(gaussians._xyz)
            if args.add_snow:
                if args.overlay_fall:
                    gaussians.create_flake(flake_units, radi=50)
                else:
                    flake_units = args.fall
                    sub_division = 5000
                    snow_list, num_list = init_snow_gaussians(flake_units, sub_division, sky_color, radi=200)
                    for i in range(len(snow_list)):
                        snow_gc = snow_list[i]
                        flake_number = num_list[i]
                        
                        snow_fall = gen_snowfall_vector(flake_number, fall_speed=0.65, gravity=None)
                        # snow_gc._xyz -= snow_fall * 100
                    
                
            if args.add_rain:
                droplets = args.fall
                sub_division = 10000
                rain_list, num_list, rot_list = init_rain_gaussians(droplets, sub_division, sky_color, gravity=gravity_vec, radi=100)
                for i in range(len(rain_list)):
                    # get elements from weather gaussian list
                    rain_gc = rain_list[i]
                    flake_number = num_list[i]
                    rain_fall = gen_rainfall_vector(rot_list, flake_number, i, fall_speed=0.85, gravity=None)
                    # rain_gc._xyz += rain_fall * 140

        if args.add_snow and not args.overlay_fall:
            rendering = stylization(rendering, rendering_depth, abs_rendering_depth, style_type="snow", sky_color=sky_color)
            luminance_image = (rendering.unsqueeze(0) * weights).sum(dim=1)
            luminance_image = luminance_image.squeeze(0)
            rendering = rendering.permute(1, 2, 0)
            nosky_mask = (rendering_depth < 0.9).all(dim=-1)
            sky_mask = rendering_depth > 0.9
            if abs_rendering_depth[nosky_mask].numel()!=0:
                rd_max = abs_rendering_depth[nosky_mask].max()
            else:
                rd_max = abs_rendering_depth.max()
            for i in range(len(snow_list)):
                snow_gc = snow_list[i]
                flake_number = num_list[i]
                
                snow_fall = gen_snowfall_vector(flake_number, fall_speed=0.65, gravity=None)
                snow_gc._xyz += snow_fall

                snow_pkg = render(view, snow_gc, pipeline, background, cam_type=cam_type)
                snow_rendering = snow_pkg["render"]
                snow_depth = snow_pkg["depth"]
                snow_rendering = snow_rendering.permute(1, 2, 0)
                luminance_factor = cal_luminance_factor(luminance_image, sky_brightness, "snow")
                snow_depth = restore_depth_fast(snow_depth, 9)
                ex_snow_depth = snow_depth.expand(3,-1,-1).permute(1, 2, 0)
                ex_abs_rendering_depth = abs_rendering_depth.expand(3,-1,-1).permute(1, 2, 0)
                sd_max = ex_snow_depth.max()


                mask = ((snow_rendering > 0.2) &
                        ((ex_snow_depth/sd_max) < (ex_abs_rendering_depth/rd_max)) &
                        (ex_snow_depth/sd_max < 0.3)).any(dim=-1)
                snow_rendering *= luminance_factor
                rendering[mask] += snow_rendering[mask]
            rendering = rendering.permute(2, 0, 1)

        elif args.add_rain:
            rendering = stylization(rendering, rendering_depth, abs_rendering_depth, style_type="rain", sky_color=sky_color)
            luminance_image = (rendering.unsqueeze(0) * weights).sum(dim=1)
            luminance_image = luminance_image.squeeze(0)
            rendering = rendering.permute(1, 2, 0)
            nosky_mask = (rendering_depth < 0.9).all(dim=-1)
            if abs_rendering_depth[nosky_mask].numel()!=0:
                rd_max = abs_rendering_depth[nosky_mask].max()
            else:
                rd_max = abs_rendering_depth.max()

            for i in range(len(rain_list)):
                # get elements from weather gaussian list
                rain_gc = rain_list[i]
                flake_number = num_list[i]

                rain_fall = gen_rainfall_vector(rot_list, flake_number, i, fall_speed=1.0, gravity=None)
                # rain_fall = gen_rainfall_vector(rot_list, flake_number, i, fall_speed=0.85, gravity=None)
                rain_gc._xyz -= rain_fall

                rain_pkg = render(view, rain_gc, pipeline, background, cam_type=cam_type)
                rain_rendering = rain_pkg["render"]
                rain_depth = rain_pkg["depth"]

                rain_rendering = rain_rendering.permute(1, 2, 0)
                luminance_factor = cal_luminance_factor(luminance_image, sky_brightness, "rain")

                rain_depth = restore_depth_fast(rain_depth, 9)
                # print(snow_depth.shape)
                ex_rain_depth = rain_depth.expand(3, -1, -1).permute(1, 2, 0)
                ex_abs_rendering_depth = abs_rendering_depth.expand(3, -1, -1).permute(1, 2, 0)
                sd_max = ex_rain_depth.max()

                mask = ((rain_rendering > 0.0) &
                        ((ex_rain_depth / sd_max) < (ex_abs_rendering_depth / rd_max)) &
                        (ex_rain_depth / sd_max < 0.8)).all(dim=-1)
                rain_rendering *= luminance_factor
                rendering[mask] += rain_rendering[mask]

            rendering = rendering.permute(2, 0, 1)

        # Static style edit
        if args.add_style:
            rendering_style = stylization(rendering, rendering_depth, abs_rendering_depth, style_type=args.style_type,
                                        style_intensity=args.intensity)


        
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        render_depths.append(to8b(rendering_depth).transpose(1, 2, 0))
        render_list.append(rendering)
        depth_list.append(rendering_depth)
        normal_list.append(normal_map)
        if args.add_style:
            style_list.append(rendering_style)
            render_styles.append(to8b(rendering_style).transpose(1, 2, 0))

        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)

        # index += 1.0

    time2 = time()
    print("FPS:", (len(views) - 2) / (time2 - time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    multithread_write(normal_list, normal_path)

    multithread_write(depth_list, depth_path)
    if args.add_style:
        multithread_write(style_list, style_path)

        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_style.mp4'), render_styles,
                            fps=30)

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images,
                    fps=30)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_depth.mp4'), render_depths,
                    fps=30)


def render_sets(dataset: ModelParams, hyperparam, iteration: int, pipeline: PipelineParams, skip_train: bool,
                skip_test: bool, skip_video: bool, interpolate: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        bg_color = [0, 0, 0]
        # bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene.gaussians, pipeline,
                       background, cam_type, scene.model_path, interpolate)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene.gaussians, pipeline,
                       background, cam_type, scene.model_path, interpolate)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), scene.gaussians, pipeline,
                       background, cam_type, scene.model_path, interpolate)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--interpolate", default=0, type=int)
    parser.add_argument("--intensity", default=1.0, type=float)
    parser.add_argument("--add_style", default=False, type=bool)
    parser.add_argument("--style_type", default='fog', type=str)
    parser.add_argument("--add_snow", default=False, type=bool)
    parser.add_argument("--add_rain", default=False, type=bool)
    parser.add_argument("--fall", default=1000001, type=int)
    parser.add_argument("--acc_snow", default=False, type=bool)
    parser.add_argument("--acc_depth", default=False, type=bool)
    parser.add_argument("--add_shadow", default=False, type=bool)
    parser.add_argument("--overlay_fall", default=False, type=bool)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams

        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train,
                args.skip_test, args.skip_video, args.interpolate)
