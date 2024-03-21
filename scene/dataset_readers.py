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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, OctreeGaussian, OctreeGaussianNode
import numpy as np
import json
import laspy
import subprocess
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.system_utils import mkdir_p
from scene.gaussian_model import BasicPointCloud
from scene.potree_loader import loadPotree

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float
    cy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[ Scene ] Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        cx = width / 2.0
        cy = height / 2.0

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "[ ERROR ] Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, cx = cx, cy = cy)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if vertices.__contains__('red'):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.zeros_like(positions)

    if vertices.__contains__('nx'):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def storeLas(path, xyz, rgb):

    # 1. Create a Las
    header = laspy.LasHeader(point_format=2, version="1.4")

    # color_att = ["R", "G", "B"]
    # for c in color_att:
    #     header.add_extra_dim(laspy.ExtraBytesParams(name=c, type=np.float32))

    # 2. Create a Las
    out_las = laspy.LasData(header)

    # 3. Fill the Las
    out_las.x = xyz[:, 0]
    out_las.y = xyz[:, 1]
    out_las.z = xyz[:, 2]

    def normalize_color(color):
        return (color * 255).astype(np.uint16)

    out_las.red = normalize_color(rgb[:, 0])
    out_las.green = normalize_color(rgb[:, 1])
    out_las.blue = normalize_color(rgb[:, 2])

    # Fill the color attributes
    # for c in color_att:
    #     setattr(out_las, c, rgb[:, color_att.index(c)])

    # Save the LAS file
    out_las.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("[ Dataloader ] Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    else:
        print("[ Dataloader ] Found .ply point cloud, skipping conversion.")
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        if "aabb" in contents:
            aabb = contents["aabb"]
            scale = max(0.000001,max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
                            abs(float(aabb[1][1])-float(aabb[0][1]))),
                            abs(float(aabb[1][2])-float(aabb[0][2]))))

            offset = [((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) - 0.5 * scale,
                ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) - 0.5 * scale, 
                ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5)- 0.5 * scale]

        elif "scale" in contents and "offset" in contents:
            scale = 1.0 / contents["scale"]
            offset = -np.array(contents["offset"]) * scale
        else:
            scale = 2.6 # default scale for NeRF scenes
            offset = -1.3

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, unit=" images", desc=f"Loading Images")):
            cam_name = os.path.join(path, frame["file_path"])
            if not os.path.exists(cam_name):
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if "cx" in contents and "cy" in contents:
                cx = contents["cx"]
                cy = contents["cy"]
            else:
                cx = image.size[0] / 2
                cy = image.size[1] / 2
                
            # Extract focal length the transform matrix
            if 'camera_angle_x' in contents or 'camera_angle_y' in contents:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = fov2focal(contents["camera_angle_x"], 2 * cx) if 'camera_angle_x' in contents else None
                fl_y = fov2focal(contents["camera_angle_y"], 2 * cy) if 'camera_angle_y' in contents else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x
                FovX = fovx = focal2fov(fl_x, 2 * cx)
                FovY = fovy = focal2fov(fl_y, 2 * cy)
            elif 'fl_x' in contents or 'fl_y' in contents:
                fl_x = (contents['fl_x'] if 'fl_x' in contents else contents['fl_y'])
                fl_y = (contents['fl_y'] if 'fl_y' in contents else contents['fl_x'])
                FovX = fovx = focal2fov(fl_x, 2 * cx)
                FovY = fovy = focal2fov(fl_y, 2 * cy)
            elif 'K' in frame or 'intrinsic_matrix' in frame:
                K = frame['K'] if 'K' in frame else frame['intrinsic_matrix']
                FovX = fovx = focal2fov(K[0][0], 2.0*K[0][2])
                FovY = fovy = focal2fov(K[1][1], 2.0*K[1][2])
                cx, cy = K[0][2], K[1][2]
            elif 'focal_length' in frame:
                FovX = fovx = focal2fov(frame['focal_length'], 2 * cx)
                FovY = fovy = focal2fov(frame['focal_length'], 2 * cy)
            else:
                raise Exception("[ ERROR ] No camera intrinsics found in the transforms file.")
                
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], cx = cx, cy = cy))
            
    return cam_infos, scale, offset

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # Read Train Cameras
    if os.path.exists(os.path.join(path, "transforms_train.json")):
        print("[ Dataloader ] Reading Training Transforms From: transforms_train.json")
        train_cam_infos, scale, offset = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    elif os.path.exists(os.path.join(path, "transforms.json")):
        print("[ Dataloader ] Reading Training Transforms From: transforms.json")
        train_cam_infos, scale, offset = readCamerasFromTransforms(path, "transforms.json", white_background, extension)

    # Read Test Cameras
    if os.path.exists(os.path.join(path, "transforms_test.json")) and eval:
        print("[ Dataloader ] Reading Test Transforms From: transforms_test.json")
        test_cam_infos, scale, offset = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Read Point Cloud
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"[ Dataloader ] Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * scale + offset
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readPotreeColmapInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    octree_path = os.path.join(path, "octree")
    las_path = os.path.join(path, "octree/0/points3D.las")
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    pcd = None
    if not os.path.exists(las_path):
        print("[ Dataloader ] Converting point3d.bin to LOD PCS, will happen only the first time you open the scene.")
        if not os.path.exists(ply_path):
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        else:
            print("[ Dataloader ] Found .ply point cloud, skipping conversion.")
            pcd = fetchPly(ply_path)
            xyz, rgb = pcd.points, pcd.colors

        print("[ Dataloader ] Converting points to LAS format.")
        mkdir_p(os.path.dirname(las_path))
        storeLas(las_path, xyz, rgb)

        # Convert to octree
        print("[ Dataloader ] Converting LAS to octree for level-of-detail pointclouds.")
        command = [os.path.join(os.getcwd(), "PotreeConverter/bin/RelWithDebInfo/Converter.exe"), las_path, "-o", octree_path, "--overwrite"]
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[ Dataloader ] LOD pointclouds generated.")

    else:
        print("[ Dataloader ] Found octree, skipping conversion.")

    # init the potree scene
    # potree = OctreeGaussian(octree_path)
        
    # print(octree_path)
        
    potree = loadPotree(octree_path)

    print(potree)

    print(potree.root.numGaussians)
    # print(potree.pointAttributes["position"]["buffer"])
    print(potree.root.octreeGeometry.pointAttributes["position"]["buffer"])

    print(len(potree.root.octreeGeometry.pointAttributes["position"]["buffer"]))

    # print(potree.root.children[0].numGaussians)
    # print(potree.root.children[1].numGaussians)

    # name r01 04 05 06
    # print(potree.root.children[0].children[5].children)

    # test with construct ply from potree
    def collect_position_buffers(node):
        position_buffers = []

        if hasattr(node, 'octreeGeometry') and "position" in node.octreeGeometry.pointAttributes:
            position_buffer = node.octreeGeometry.pointAttributes["position"]["buffer"]
            position_array = np.array(position_buffer, dtype=np.float32).reshape(-1, 3)  # 每三个数为一组，转换为二维数组
            position_buffers.append(position_array)
            # position_buffers.append(position_buffer)

        if hasattr(node, 'children'):
            for child in node.children:
                if child is not None:
                    position_buffers.extend(collect_position_buffers(child))

        return position_buffers

    # recursive read the potree
    position_buffers = collect_position_buffers(potree.root)

    all_positions = np.concatenate(position_buffers, axis=0)
    print(all_positions.shape)

    # rearrange buffer and dump to ply
    output_path = r"D:\workspace\mipnerf360\bicycle_lod\octree\point_cloud_from_potree.ply"
    vertices = np.array(
        [(position[0], position[1], position[2]) for position in all_positions],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(output_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    assert 0, "[ Debugging ] Potree scenes are not supported yet!"

    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Potree" : readPotreeColmapInfo
}