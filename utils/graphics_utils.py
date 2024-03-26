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
import math
import numpy as np
from typing import NamedTuple

# simple vector3
class Vector3:

    def __init__(self, x=float(0), y=float(0), z=float(0)):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"
    
    def set(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def addVectors(self, vecA: 'Vector3', vecB: 'Vector3') -> 'Vector3':
        self.x = vecA.x + vecB.x
        self.y = vecA.y + vecB.y
        self.z = vecA.z + vecB.z
        return self
    
    def subVectors(self, vecA: 'Vector3', vecB: 'Vector3') -> 'Vector3':
        self.x = vecA.x - vecB.x
        self.y = vecA.y - vecB.y
        self.z = vecA.z - vecB.z
        return self
    
    def multiplyScalar(self, scalar: float) -> 'Vector3':
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        return self

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> 'Vector3':
        l = self.length()
        if l > 0:
            return self / l
        return self

# simple bbox
class BoundingBox:

    def __init__(self, min_point: Vector3 = Vector3(), max_point: Vector3 = Vector3()) -> None:
        self.min = min_point
        self.max = max_point

    def __repr__(self) -> str:
        return f"BoundingBox({self.min}, {self.max})"
    
    def isEmpty(self) -> bool:
        return (self.max.x < self.min.x) or (self.max.y < self.min.y) or (self.max.z < self.min.z) 

    def getCenter(self) -> Vector3:
        return Vector3(0, 0, 0) if self.isEmpty() else (self.min + self.max) * 0.5
    
    def getSize(self) -> Vector3:
        return Vector3(0, 0, 0) if self.isEmpty() else (self.max - self.min)

    def getBoundingSphere(self):
        sphere = BoundingSphere()
        sphere.set(center=self.getCenter(), radius=self.getSize().length() * 0.5)
        return sphere

# simple bounding sphere   
class BoundingSphere:

    def __init__(self, center=Vector3(), radius=0.0) -> None:
        self.center = center
        self.radius = radius

    def __repr__(self) -> str:
        return f"BoundingSphere({self.center}, {self.radius})"

    def set(self, center: Vector3, radius: float) -> None:
        self.center = center
        self.radius = radius

class OctreeGaussian:
    def __init__(self):
        self.spacing = 0
        self.boundingbox = None
        self.root = None
        self.scale = None
        self.pointAttributes = None
        self.loader = None
        self.maxLevel = 0

class OctreeGaussianNode:
    # Static variables
    IDCount = 0

    def __init__(self, name, octreeGaussian, boundingbox: BoundingBox):
        self.id = OctreeGaussianNode.IDCount
        OctreeGaussianNode.IDCount += 1
        self.name = name
        self.index = 0 if name == "r" else int(name[-1])
        self.nodeType = 0
        self.hierarchyByteOffset = 0
        self.hierarchyByteSize = 0
        self.byteOffset = 0
        self.byteSize = 0
        self.spacing = 0
        self.projection = 0
        self.offset = 0
        self.gaussian_model = None
        self.octreeGaussian = octreeGaussian
        self.pointcloud = None
        self.boundingbox = boundingbox
        self.children = [None for _ in range(8)]
        self.parent = None
        self.numGaussians = 0
        self.level = None
        self.loaded = False
        self.loading = False

    def loadGaussianData(self):
        pass
    
    def __repr__(self):
        return f"OctreeGaussianNode {self.name}"
    
    def isGeometryNode(self) -> bool:
        return True
    
    def isTreeNode(self) -> bool:
        return False
    
    def getLevel(self) -> int:
        return self.level
    
    def getBoundingSphere(self):
        return self.boundingSphere
    
    def getBoundingBox(self): 
        return self.boundingbox

    def getChildren(self):
        return self.children
    
    def getNumGaussians(self):
        return self.numGaussians

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))