from pysubdiv.main.data import files
from pysubdiv.create_control_cage import control_cage
import numpy as np

# First step: import all the necessary meshes from .obj file
original_meshes = [files.read('FaultDomain/meshes/gemp_mesh_0_scaled.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_1_cut.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_1_cut_2.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_2_cut_1.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_2_cut_2.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_3_cut_1.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_3_cut_2.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_4_cut_1.obj'),
                   files.read('FaultDomain/meshes/gemp_mesh_4_cut_2.obj')]
# Second step load the module control_cage: it will take our list of original meshes as the first parameter, then
# we can pass additional keyword arguments to let the algorithm compute some critical vertex positions for us.
# The function control cage will return a mesh, so we need to store a reference in a variable:
# find_vertices will find vertices lying on the boundary of the input meshes and on the corner and displays them in
# the viewer. calc_intersection tries to find vertices where two planes intersect, as the input meshes don't have
# an intersection it is here set to False. add_boundary_box will calculate the four corner vertices of the meshes'
# boundary box. This is here necessary to form a watertight mesh in the end. use_dynamic_faces will allow us to set
# the vertices and faces of the control cage as static or dynamic. This constrains will later help in the approximation.

control_cage = control_cage.create_control_cage(original_meshes, find_vertices=True, simple_subdivision=True,
                                                iteration_subdivision=0)
# save the control cage obj file
control_cage.save_mesh('save_me.obj')
# save the control cage data
control_cage.save_data('save_me_data')

# load a mesh
faultDomainControlCage = files.read('FaultDomain/controlCage/FaultDomainControlCage.obj')
# load data of the mesh
faultDomainControlCage.load_data('FaultDomain/controlCage/FaultDomainControlCageData.pkl')

faultDomainControlCage.visualize_mesh()

# print array of dynamic faces, the index of the array corresponds to the face and the element is
# the index of the mesh the face should be fitted to 's' means static no fit
print("Dynamic faces: \n", faultDomainControlCage.data['dynamic_faces'])
# print array of dynamic vertices, the index of the array corresponds to the vertex and the element is
# the index of the mesh the vertex should be fitted to 's' means static -> no fit

print("Dynamic vertices: \n", faultDomainControlCage.data['dynamic_vertices'])

# find indices of vertices to be fitted to mesh_3 with numpy.nonzero -> return indices where condition is true
# Caution: mesh indices are stored as strings
indices_vertices_mesh_3 = np.nonzero(faultDomainControlCage.data['dynamic_vertices'] == '3')
print("Index of dynamic vertices fitted to mesh_3: \n", indices_vertices_mesh_3)
# When doing subdivision these properties are inherited to the subdivided mesh
faultDomainControlCage_simple_subdivided = faultDomainControlCage.simple_subdivision(2)
print("Dynamic vertices: \n", faultDomainControlCage_simple_subdivided.data['dynamic_vertices'])
