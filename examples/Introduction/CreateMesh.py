import PySubdiv as psd
from PySubdiv.data import files

import os
absolute_path = os.path.dirname(__file__)


# Create a simple Mesh by passing vertices and faces
# Create empty object or pass vertices and faces directly
mesh = psd.Mesh()


# vertices do have the shape (n,3), we can pass a list or numpy array
# where n is the number of vertices
# each vertex is defined by its x,y and z coordinate in space

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
