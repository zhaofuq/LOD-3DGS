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
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, warpped_depth
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    # Set up output folder
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join(dataset.source_path, "3D-Gaussian-Splatting", unique_str[0:10])
    print("[ Training ] Output Folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)

    # Load dataset
    scene = Scene(dataset)
    scene.training_setup(opt)

    # extract scene depth max
    dataset.depth_max = scene.depth_max.cpu().item()

    # prepare logger and extract parameters
    tb_writer = prepare_output_and_logger(dataset)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        scene.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    level_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    xyz, features, opacity, scales, rotations, cov3D_precomp, \
                        active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(viewpoint_cam.world_view_transform, pipe.compute_cov3D_python, scaling_modifer)
                    net_image = render(custom_cam,  xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipe, background, scaling_modifer, cov3D_precomp = cov3D_precomp)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        scene.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            scene.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Pick a random Level
        if not level_stack:
            level_stack = list(range(-scene.max_level, scene.max_level + 1))
        random_level = level_stack.pop(randint(0, len(level_stack)-1))
        # random_level =-1
  
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        xyz, features, opacity, scales, rotations, cov3D_precomp, \
            active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(viewpoint_cam.world_view_transform, pipe.compute_cov3D_python, random = random_level)
        render_pkg = render(viewpoint_cam,  xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipe, background, cov3D_precomp = cov3D_precomp)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        depth = warpped_depth(depth)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        depth_loss = 0.0

        if viewpoint_cam.depth is not None:
            gt_depth = viewpoint_cam.depth.cuda()
            depth_loss = 2.0 * l2_loss(depth, gt_depth)
   
        loss = rgb_loss + depth_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0.0
            if viewpoint_cam.depth is not None:
                ema_depth_loss_for_log = 0.4 * depth_loss.item()  + 0.6 * ema_depth_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"RGB Loss": f"{ema_loss_for_log:.{4}f}", "Depth Loss": f"{ema_depth_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, rgb_loss, depth_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("[ Training ] [ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                scene.update_max_radii2D(radii, visibility_filter, masks)
                scene.add_densification_stats(viewspace_point_tensor, visibility_filter, masks)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    scene.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    scene.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimizer_step()

def prepare_output_and_logger(args):    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[ Training ] Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, rgb_loss, depth_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/rgb_loss', rgb_loss.item(), iteration)
        if isinstance(depth_loss, torch.Tensor):
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', rgb_loss.item() + depth_loss.item(), iteration)
        else:
            tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', rgb_loss.item() + depth_loss, iteration)
        
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('memory/memory_allocated', torch.cuda.memory_allocated('cuda') / (1024 ** 3), iteration)
        tb_writer.add_scalar('memory/memory_reserved', torch.cuda.memory_reserved('cuda') / (1024 ** 3), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                l1_test, psnr_test = [], []
                for idx, viewpoint in enumerate(config['cameras']):
                    xyz, features, opacity, scales, rotations, cov3D_precomp, \
                        active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(viewpoint.world_view_transform, renderArgs[0].compute_cov3D_python)
                    results = renderFunc(viewpoint, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, cov3D_precomp = cov3D_precomp, *renderArgs)
                    image = torch.clamp(results["render"], 0.0, 1.0)
                    depth = results["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if viewpoint.depth is not None:
                        gt_depth = viewpoint.depth.to("cuda")
                        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
                    l1_test.append(l1_loss(image, gt_image))
                    psnr_test.append(psnr(image, gt_image).mean())
                    if tb_writer and (idx == 0):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image.unsqueeze(0), global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth.unsqueeze(0), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image.unsqueeze(0), global_step=iteration)
                            if viewpoint.depth is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth_depth".format(viewpoint.image_name), gt_depth.unsqueeze(0), global_step=iteration)
                for level in range(scene.max_level + 1):
                    viewpoint = config['cameras'][0]  #[randint(0, len(config['cameras'])-1)]
                    xyz, features, opacity, scales, rotations, cov3D_precomp, \
                        active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(viewpoint.world_view_transform, renderArgs[0].compute_cov3D_python, random=level)
                    image = torch.clamp(renderFunc(viewpoint, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, cov3D_precomp = cov3D_precomp, *renderArgs)["render"], 0.0, 1.0)
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/level_{}".format(viewpoint.image_name, level), image.unsqueeze(0), global_step=iteration)

                l1_test = sum(l1_test) / len(l1_test)
                psnr_test = sum(psnr_test) / len(psnr_test)     
                print("\n[ Training ] [ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.getGaussians().get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.getGaussians().get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50, 100, 500, 1_000, 5_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("[ Training ] Optimizing With Parameters: " + str(vars(args)))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("[ Training ] Training complete.")