
# PySubdiv 
Open-source python package for non-manifold subdivison surfaces algorithm

# Overview
The PySubdiv library is a python-based open-source package that can apply non-manifold subdivision surfaces algorithm (Loop) on triangluar meshes with considering unique semi-sharp creases value for each edge. The implementation of semi-sharp creases in the algorithm allows to model of different varieties of structures with sharp edges. Fitting of smooth subdivision surfaces to the input dense meshes for reconstruction is implemented in this python package.

# Installation

Tested on Ubuntu 22.04.1 

1. Clone the repository from source (We are currently working with a pip installation)
```console
git clone https://github.com/SimBe-hub/PySubdiv.git
```

2. Install the packages and dependencies
```console
cd PySubdiv
python3 setup.py install
```
for developers
```console
python3 setup.py develop
```
3. **Important：**
install `easygui` from apt, NOT pip

```console
apt-get install python3-easygui
```

Test if the installation is successful

```console
python3 examples/Introduction/CreateMesh.py
```


# Quick Start




```python
import PySubdiv as psd
from PySubdiv.data import files

import os
absolute_path = os.path.dirname(__file__)

# Create a simple Mesh by passing vertices and faces
# Create empty object or pass vertices and faces directly
mesh = psd.Mesh()

# vertices do have the shape (n,3), we can pass a list or numpy array
# where n is the number of vertices
# each vertex is defined by it's x,y and z coordinate in space

vertices = [[-3., 3., -7.], [1., 2., 1.], [1., 3., -7.], [-3., -1., 1.], [1., -1., 1.], [-8., -1., -2.],
            [-12., -1., -8.], [-8., -1., -10.], [1., -1., -7.], [-3., -1., -7.], [-8., 2., -2.], [-12., 3., -8.],
            [-12., 2., 0.], [-3., 2., 1.], [-12., -1., 0.], [-8., 3., -10.]]
# set the vertices property
mesh.vertices = vertices

# Faces are defined by the indices of the vertices and have the shape (n,3) for triangular meshes.
# n is the number if faces. Each face is defined by three vertices


faces = [[0, 1, 2], [1, 3, 4], [5, 6, 7], [8, 3, 9], [8, 1, 4], [0, 8, 9],
         [10, 11, 12], [0, 10, 13], [10, 14, 5], [7, 11, 15], [6, 12, 11],
         [9, 5, 7], [13, 5, 3], [9, 15, 0], [0, 13, 1], [1, 13, 3], [5, 14, 6],
         [8, 4, 3], [8, 2, 1], [0, 2, 8], [10, 15, 11], [0, 15, 10], [10, 12, 14],
         [7, 6, 11], [6, 14, 12], [9, 3, 5], [13, 10, 5], [9, 7, 15]]
# set the faces property
mesh.faces = faces
# visualize the mesh
mesh.visualize_mesh()

# As it is quite tedious to set vertices and faces manually we simply can load and also save meshes from and to obj file
# load mesh by passing the file path as string
relative_path = "Meshes/CoarseChannel.obj"
full_path = os.path.join(absolute_path, relative_path)
mesh_loaded_from_obj = files.read(full_path)

# lets have a look:
mesh_loaded_from_obj.visualize_mesh()
# Yes indeed it is the same mesh
# saving the mesh to obj file by passing a permitted file path, here I'm just overwriting.
# The file ending will be added automatically if forgotten.
mesh.save_mesh(full_path)
```

# References

Please cite this article when you want to use this code.

Moulaeifard, M., Bernard, S., and Wellmann, F.: PySubdiv: open-source geological modelling and reconstruction by non-manifold subdivision surfaces, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2022-685, 2022.

# License

The PySubdiv is under GNU license. https://zenodo.org/record/6878051#.YuGVynZByUk


# Developers

Full stack developer : Simon Bernard (SimBe-hub)

Project manager and backend developer : Mohammad Moulaeifard (MohammadCGRE)

Project supervisor from RWTH Aachen university: Prof. Florian Wellmann
