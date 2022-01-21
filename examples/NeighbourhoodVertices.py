from pysubdiv.main.data import files
import numpy as np
# find connected vertices
# load the mesh
cube = files.read("meshes/simple_cube.obj")
#cube.visualize_mesh()

# create vertex adjacency matrix
cube.vertices_connected()
# print the matrix by calling the dictionary with the correct key
print(cube.data['vertices_adjacency_matrix'])
# create vertex adjacency list
vertex_adjacency_list = cube.vertices_connected_to_list()
# print the matrix by calling the dictionary with the correct key
print(cube.data['vertices_adjacency_list'])

# average position of adjacent vertices for vertex with index zero
print(cube.vertices[vertex_adjacency_list[0]])
print(np.mean(cube.vertices[vertex_adjacency_list[0]], axis=0))
