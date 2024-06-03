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

import os
import random
import json
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, load_iteration=-1, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.level = 0
        
        if os.path.exists(os.path.join(self.model_path, "point_cloud")):
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("[ Scene ] Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.use_lod:
                print("[ Scene ] Found sparse folder, assuming Octree data set!")
                scene_info = sceneLoadTypeCallbacks["Octree"](args.source_path, args.images, args.depths, args.eval)
            else:
                print("[ Scene ] Found sparse folder, assuming Colmap data set!")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("[ Scene ] Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("[ Scene ] Found transforms.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.max_level = scene_info.max_level
        self.beta = np.log(self.max_level+1)
        self.gaussians = [GaussianModel(args.sh_degree, level) for level in range(self.max_level + 1)]
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("[ Scene ] Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("[ Scene ] Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        self.depth_min = torch.tensor(float('inf'))
        self.depth_max = torch.tensor(-float('inf'))
        # Load Gaussian Model
        import time
        st = time.time()
        for level in range(self.max_level + 1):
            if self.loaded_iter:
                self.gaussians[level].load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "level_{}.ply".format(level)))
            else:
                self.gaussians[level].create_from_pcd(scene_info.point_cloud[level], self.cameras_extent)

            for cam in self.train_cameras[1.0]:
                xyz = self.gaussians[level].get_xyz.detach()
                depth_z = self.get_z_depth(xyz, cam.world_view_transform)
                self.depth_min = torch.min(self.depth_min, torch.max(depth_z.min(), torch.tensor(0.0)))
                self.depth_max = torch.max(self.depth_max, depth_z.max())
        self.depth_max = 0.95 * 1.3 * (self.depth_max - self.depth_min) + self.depth_min
        self.depth_min = 0.05 * 1.3 * (self.depth_max - self.depth_min) + self.depth_min
        print("[ Scene ] Initialize scene depth range at [{:2f}, {:2f}]".format(self.depth_min.cpu(), self.depth_max.cpu()))
        et = time.time()
        print("[ Scene ] Gaussian Model creation took {} seconds".format(et - st))
        

    def get_z_depth(self, xyz, viewmatrix):
        homogeneous_xyz = torch.cat((xyz, torch.ones(xyz.shape[0], 1, dtype=xyz.dtype, device=xyz.device)), dim=1)
        projected_xyz= torch.matmul(homogeneous_xyz, viewmatrix)
        depth_z = projected_xyz[:,2]
        return depth_z

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if self.max_level == 0:
            self.save_full_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:   
            for level in range(self.max_level+1):
                self.gaussians[level].save_ply(os.path.join(point_cloud_path, "level_{}.ply".format(level)))
        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.gaussians[-1]._features_dc.shape[1]*self.gaussians[-1]._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.gaussians[-1]._features_rest.shape[1]*self.gaussians[-1]._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.gaussians[-1]._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.gaussians[-1]._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_full_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz, f_dc, f_rest, opacity, scales, rotations = [], [], [], [], [], []
        for level in range(self.max_level + 1):
            xyz.append(self.gaussians[level]._xyz)
            f_dc.append(self.gaussians[level]._features_dc)
            f_rest.append(self.gaussians[level]._features_rest)
            opacity.append(self.gaussians[level]._opacity)
            scales.append(self.gaussians[level]._scaling)
            rotations.append(self.gaussians[level]._rotation)

        xyz = torch.cat(xyz, dim=0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat(f_dc, dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = torch.cat(f_rest, dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = torch.cat(opacity, dim=0).detach().cpu().numpy()
        scale = torch.cat(scales, dim=0).detach().cpu().numpy()
        rotation = torch.cat(rotations, dim=0).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getGaussians(self, level=-1):
        return self.gaussians[level]
    
    def getLevels(self):
        return self.max_level
    
    def update_max_radii2D(self, radii, visibility_filter, masks):
        level_start = 0
        expanded_visibility_filter = torch.zeros(masks.shape[0], dtype=torch.bool, device=visibility_filter.device)
        expanded_radii = torch.zeros(masks.shape[0], dtype=radii.dtype, device=radii.device)
        expanded_visibility_filter[masks] = visibility_filter
        expanded_radii[masks] = radii
        for level in range(self.max_level + 1):
            level_offset = self.gaussians[level].max_radii2D.shape[0]
            level_radii = expanded_radii[level_start:level_start+level_offset]
            level_visibility_filter = expanded_visibility_filter[level_start:level_start+level_offset]
            self.gaussians[level].max_radii2D[level_visibility_filter] = torch.max(self.gaussians[level].max_radii2D[level_visibility_filter], level_radii[level_visibility_filter])
            level_start += level_offset

    def training_setup(self, args):
        for level in range(self.max_level + 1):
            self.gaussians[level].training_setup(args)

    def restore(self, params, args):
        for level in range(self.max_level + 1):
            self.gaussians[level].restore(params, args)
    
    def update_learning_rate(self, iters):
        for level in range(self.max_level + 1):
            self.gaussians[level].update_learning_rate(iters)
        
    def oneupSHdegree(self):
        for level in range(self.max_level + 1):
            self.gaussians[level].oneupSHdegree()
    
    def add_densification_stats(self, viewspace_point, visibility_filter, masks):
        level_start = 0
        viewspace_point_grad = viewspace_point.grad
        expanded_viewspace_point_grad = torch.zeros(masks.shape[0], 3, dtype=viewspace_point_grad.dtype, device=viewspace_point_grad.device)
        expanded_visibility_filter = torch.zeros(masks.shape[0], dtype=torch.bool, device=visibility_filter.device)
        expanded_viewspace_point_grad[masks,:] = viewspace_point_grad
        expanded_visibility_filter[masks] = visibility_filter
        for level in range(self.max_level + 1):
            level_offset = self.gaussians[level].get_xyz.shape[0]
            level_viewspace_point_grad = expanded_viewspace_point_grad[level_start:level_start + level_offset]
            level_visibility_filter = expanded_visibility_filter[level_start:level_start + level_offset]
            self.gaussians[level].add_densification_stats(level_viewspace_point_grad, level_visibility_filter)    
            level_start += self.gaussians[level].get_xyz.shape[0]

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        for level in range(self.max_level + 1):
            scale = np.min([np.sqrt(2) ** (self.max_level - level), 4.0])   #np.log(self.max_level - level + 1.0) + 1.0
            if max_screen_size:
                max_screen_size = max_screen_size * scale
            self.gaussians[level].densify_and_prune(max_grad * scale, min_opacity, extent * scale, max_screen_size)
    
    def reset_opacity(self):
        for level in range(self.max_level + 1):
            self.gaussians[level].reset_opacity()

    def optimizer_step(self):
        for level in range(self.max_level + 1):
            self.gaussians[level].optimizer.step()
            self.gaussians[level].optimizer.zero_grad(set_to_none = True)


    def get_gaussian_parameters(self, viewpoint, compute_cov3D_python, scaling_modifier=1.0, random=-1):

        levels = range(self.max_level + 1)
        get_attrs = lambda attr: [getattr(self.gaussians[level], attr) for level in levels]
        xyz, features, opacity, scales, rotations = map(get_attrs, ['get_xyz', 'get_features', 'get_opacity', 'get_scaling', 'get_rotation'])

        # Compute cov3D_precomp if necessary
        cov3D_precomp = [self.gaussians[-1].get_covariance(scaling_modifier)] * len(xyz) if compute_cov3D_python else None

        # Define activation levels based on 'random' parameter
        if random < 0:
            depths = [self.get_z_depth(xyz_lvl.detach(), viewpoint) for xyz_lvl in xyz]
            act_levels = [torch.clamp((self.max_level + 1) * torch.exp(-1.0 * self.beta * torch.abs(depth) / self.depth_max), 0, self.max_level) for depth in depths]
            act_levels = [torch.floor(level) for level in act_levels]
            filters = [act_level == level for act_level, level in zip(act_levels, levels)]
        else:
            filters = [torch.full_like(xyz[level][:,0], level == random, dtype=torch.bool) for level in levels]

        # Concatenate all attributes
        concat_attrs = lambda attrs: torch.cat(attrs, dim=0)
        xyz, features, opacity, scales, rotations, filters = map(concat_attrs, [xyz, features, opacity, scales, rotations, filters])

        # Apply filters to all attributes
        filtered = lambda attr: attr[filters]
        xyz, features, opacity, scales, rotations = map(filtered, [xyz, features, opacity, scales, rotations])

        if compute_cov3D_python:
            cov3D_precomp = filtered(concat_attrs(cov3D_precomp))

        # Active and maximum spherical harmonics degrees
        active_sh_degree, max_sh_degree = self.gaussians[-1].active_sh_degree, self.gaussians[-1].max_sh_degree

        return xyz, features, opacity, scales, rotations, cov3D_precomp, active_sh_degree, max_sh_degree, filters
