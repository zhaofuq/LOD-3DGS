# ref: potree\src\modules\loader\2.0\OctreeGeometry.js
# create by Penghao Wang

import os
import numpy as np
import json

from utils.graphics_utils import OctreeGaussianNode, Vector3, BoundingBox

potreeConst = {
    "pointBudget": 1 * 1000 * 1000,
    "framenumber" : 0,
    "numNodesLoading" : 0,
    "maxNodesLoading" : 4
}

class PointAttributeTypes:
    DATA_TYPE_DOUBLE = {'ordinal': 0, 'name': "double", 'size': 8}
    DATA_TYPE_FLOAT = {'ordinal': 1, 'name': "float", 'size': 4}
    DATA_TYPE_INT8 = {'ordinal': 2, 'name': "int8", 'size': 1}
    DATA_TYPE_UINT8 = {'ordinal': 3, 'name': "uint8", 'size': 1}
    DATA_TYPE_INT16 = {'ordinal': 4, 'name': "int16", 'size': 2}
    DATA_TYPE_UINT16 = {'ordinal': 5, 'name': "uint16", 'size': 2}
    DATA_TYPE_INT32 = {'ordinal': 6, 'name': "int32", 'size': 4}
    DATA_TYPE_UINT32 = {'ordinal': 7, 'name': "uint32", 'size': 4}
    DATA_TYPE_INT64 = {'ordinal': 8, 'name': "int64", 'size': 8}
    DATA_TYPE_UINT64 = {'ordinal': 9, 'name': "uint64", 'size': 8}

    @classmethod
    def as_list(cls):
        return [value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value)]

    @classmethod
    def add_dynamic_attributes(cls):
        for i, attr in enumerate(cls.as_list()):
            setattr(cls, str(i), attr)

PointAttributeTypes.add_dynamic_attributes()

class PointAttribute:
    def __init__(self, name, type, numElements):
        self.name = name
        self.type = type
        self.numElements = numElements
        self.byteSize = self.numElements * self.type['size']
        self.description = ""
        self.range = [float('inf'), float('-inf')]

# Defining the static attributes for the PointAttribute class
PointAttribute.POSITION_CARTESIAN = PointAttribute("POSITION_CARTESIAN", PointAttributeTypes.DATA_TYPE_FLOAT, 3)
PointAttribute.RGBA_PACKED = PointAttribute("COLOR_PACKED", PointAttributeTypes.DATA_TYPE_INT8, 4)
PointAttribute.COLOR_PACKED = PointAttribute.RGBA_PACKED
PointAttribute.RGB_PACKED = PointAttribute("COLOR_PACKED", PointAttributeTypes.DATA_TYPE_INT8, 3)
PointAttribute.NORMAL_FLOATS = PointAttribute("NORMAL_FLOATS", PointAttributeTypes.DATA_TYPE_FLOAT, 3)
PointAttribute.INTENSITY = PointAttribute("INTENSITY", PointAttributeTypes.DATA_TYPE_UINT16, 1)
PointAttribute.CLASSIFICATION = PointAttribute("CLASSIFICATION", PointAttributeTypes.DATA_TYPE_UINT8, 1)
PointAttribute.NORMAL_SPHEREMAPPED = PointAttribute("NORMAL_SPHEREMAPPED", PointAttributeTypes.DATA_TYPE_UINT8, 2)
PointAttribute.NORMAL_OCT16 = PointAttribute("NORMAL_OCT16", PointAttributeTypes.DATA_TYPE_UINT8, 2)
PointAttribute.NORMAL = PointAttribute("NORMAL", PointAttributeTypes.DATA_TYPE_FLOAT, 3)
PointAttribute.RETURN_NUMBER = PointAttribute("RETURN_NUMBER", PointAttributeTypes.DATA_TYPE_UINT8, 1)
PointAttribute.NUMBER_OF_RETURNS = PointAttribute("NUMBER_OF_RETURNS", PointAttributeTypes.DATA_TYPE_UINT8, 1)
PointAttribute.SOURCE_ID = PointAttribute("SOURCE_ID", PointAttributeTypes.DATA_TYPE_UINT16, 1)
PointAttribute.INDICES = PointAttribute("INDICES", PointAttributeTypes.DATA_TYPE_UINT32, 1)
PointAttribute.SPACING = PointAttribute("SPACING", PointAttributeTypes.DATA_TYPE_FLOAT, 1)
PointAttribute.GPS_TIME = PointAttribute("GPS_TIME", PointAttributeTypes.DATA_TYPE_DOUBLE, 1)

class PointAttributes:
    def __init__(self, pointAttributes=None):
        self.attributes = []
        self.byteSize = 0
        self.size = 0
        self.vectors = []

        if pointAttributes is not None:
            for pointAttributeName in pointAttributes:
                pointAttribute = getattr(PointAttribute, pointAttributeName, None)
                if pointAttribute:
                    self.attributes.append(pointAttribute)
                    self.byteSize += pointAttribute.byteSize
                    self.size += 1

    def add(self, pointAttribute):
        self.attributes.append(pointAttribute)
        self.byteSize += pointAttribute.byteSize
        self.size += 1

    def addVector(self, vector):
        self.vectors.append(vector)

    def hasNormals(self):
        normal_attributes = [
            PointAttribute.NORMAL_SPHEREMAPPED, 
            PointAttribute.NORMAL_FLOATS, 
            PointAttribute.NORMAL, 
            PointAttribute.NORMAL_OCT16
        ]
        return any(attr in self.attributes for attr in normal_attributes)

typename_typeattribute_map = {
    "double": PointAttributeTypes.DATA_TYPE_DOUBLE,
    "float": PointAttributeTypes.DATA_TYPE_FLOAT,
    "int8": PointAttributeTypes.DATA_TYPE_INT8,
    "uint8": PointAttributeTypes.DATA_TYPE_UINT8,
    "int16": PointAttributeTypes.DATA_TYPE_INT16,
    "uint16": PointAttributeTypes.DATA_TYPE_UINT16,
    "int32": PointAttributeTypes.DATA_TYPE_INT32,
    "uint32": PointAttributeTypes.DATA_TYPE_UINT32,
    "int64": PointAttributeTypes.DATA_TYPE_INT64,
    "uint64": PointAttributeTypes.DATA_TYPE_UINT64,
}

tmpVec3 = Vector3()

def createChildAABB(aabb: BoundingBox, index: int) -> BoundingBox:
    minPoint = Vector3(aabb.min.x, aabb.min.y, aabb.min.z)
    maxPoint = Vector3(aabb.max.x, aabb.max.y, aabb.max.z)
    size = tmpVec3.subVectors(maxPoint, minPoint)

    if (index & 0b0001) > 0:
        minPoint.z += size.z / 2
    else:
        maxPoint.z -= size.z / 2

    if (index & 0b0010) > 0:
        minPoint.y += size.y / 2
    else:
        maxPoint.y -= size.y / 2

    if (index & 0b0100) > 0:
        minPoint.x += size.x / 2
    else:
        maxPoint.x -= size.x / 2

    return BoundingBox(min_point=minPoint, max_point=maxPoint)

def loadPotree(path, name):
    if not os.path.exists(path):
        return None
    
    if "metadata.json" not in str(path):
        assert False, "[ Error ] Potree path does not contain metadata.json in loadPotree method"

    loadworker = potreeLoader()
    loadworker.load(path, name)

class nodeLoader():
    
    def __init__(self, path: str) -> None:
        self.path = path

    def loadHierarchy(self, node: OctreeGaussianNode) -> None:
        hierarchyByteOffset = node.hierarchyByteOffset
        hierarchyByteSize = node.hierarchyByteSize
        first = hierarchyByteOffset
        last = first + hierarchyByteSize - 1
        # load the hierarchy.bin from byte first to last
        hierarchyPath = self.path.replace("metadata.json", "hierarchy.bin")
        with open(hierarchyPath, "rb") as f:
            # load from first to last, which is index of bytes
            f.seek(first)
            buffer = f.read(last - first + 1)
        f.close()
        self.parseHierarchy(node, buffer)

    def parseHierarchy(self, node: OctreeGaussianNode, buffer):
        bytesPerNode = 22
        numNodes = len(buffer) / bytesPerNode

        octree = node.OctreeGaussian
        nodes = [None for i in range(numNodes)]
        nodes[0] = node
        nodePos = 1

        for i in range(numNodes):
            start = i * bytesPerNode
            # uint8
            type = buffer[start]
            # uint8
            childMask = buffer[start + 1]
            # uint32
            numPoints = int.from_bytes(buffer[start + 2:start + 6], byteorder='little', signed=False)
            # bigint 64
            byteOffset = int.from_bytes(buffer[start + 6:start + 14], byteorder='little', signed=True)
            # bigint 64
            byteSize = int.from_bytes(buffer[start + 14:start + 22], byteorder='little', signed=True)

            if nodes[i].nodeType == 2:
                nodes[i].byteOffset = byteOffset
                nodes[i].byteSize = byteSize
                nodes[i].numGaussians = numPoints
            elif type == 2:
                nodes[i].hierarchyByteOffset = byteOffset
                nodes[i].hierarchyByteSize = byteSize
            else:
                nodes[i].byteOffset = byteOffset
                nodes[i].byteSize = byteSize
                nodes[i].numGaussians = numPoints

            if nodes[i].byteSize == 0:
                nodes[i].numGaussians = 0

            nodes[i].nodeType = type

            if nodes[i].nodeType == 2:
                continue

            for childIndex in range(8):
                childExists = ((1 << childIndex) & childMask) != 0
                if not childExists:
                    continue

                childName = nodes[i].name + str(childIndex)
                childAABB = createChildAABB(nodes[i].boundingbox, childIndex)
                child = OctreeGaussianNode(childName, octree, childAABB)
                child.name = childName
                child.spacing = nodes[i].spacing / 2
                child.level = nodes[i].level + 1
                child.parent = nodes[i]

                nodes[i].children[childIndex] = child
                nodePos += 1

    def load(self, node: OctreeGaussianNode) -> None:

        if (node.loaded or node.loading):
            return
        
        node.loading = True
        potreeConst["numNodesLoading"] += 1
        
        if node.nodeType == 2:
            self.loadHierarchy(node=node)

        byteOffset = node.byteOffset
        byteSize = node.byteSize

        octreePath = self.path.replace("metadata.json", "octree.bin")

        first = byteOffset
        last = first + byteSize - 1

        if byteSize == 0:
            assert False, "[ Error ] byteSize is 0 in nodeLoader.load method"
        else:
            with open(octreePath, "rb") as f:
                f.seek(first)
                buffer = f.read(last - first + 1)
            f.close()



class potreeLoader():

    def __init__(self) -> None:
        self.metadata = None

    def load(self, path, name):
        if not os.path.exists(path):
            assert False, "[ Error ] Path does not exist in disk in potreeLoader.load method"
        
        # load the json file and parse it
        with open(path, 'r') as file:
            self.metadata = json.load(file)
        file.close()

        # parse the attributes
        attributes = self.parseAttributes(self.metadata["attributes"])\
        loader = NodeLoader()


    def parseAttributes(self, jsonAttributes: list) -> None:
        attributes = PointAttributes()
        replacements = {
            "rgb": "rgba"
        }
        for jsonAttribute in jsonAttributes:

            name = jsonAttribute["name"]
            description = jsonAttribute["description"]
            size = jsonAttribute["size"]
            numElements = jsonAttribute["numElements"]
            elementSize = jsonAttribute["elementSize"]
            type = jsonAttribute["type"]
            min = jsonAttribute["min"]
            max = jsonAttribute["max"]

            type = typename_typeattribute_map[type]
            potreeAttributeName = replacements[name] if name in replacements else name
            attribute = PointAttribute(potreeAttributeName, type, numElements)

            if numElements == 1:
                attribute.range = [min[0], max[0]]
            else:
                attribute.range = [min, max]

            if name == "gps-time":
                if attribute.range[0] == attribute.range[1]:
                    attribute.range[1] += 1

            # attribute.initialRange = attribute.range; # no need
            attributes.add(attribute)

        # check if it has normals
        if any(attr.name == "NormalX" for attr in attributes.attributes) and \
            any(attr.name == "NormalY" for attr in attributes.attributes) and \
            any(attr.name == "NormalZ" for attr in attributes.attributes):
            vector = {
                "name": "NORMAL",
                "attributes": ["NormalX", "NormalY", "NormalZ"]
            }
            attributes.addVector(vector)

        return attributes