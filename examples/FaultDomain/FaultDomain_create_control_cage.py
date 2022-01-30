from PySubdiv.data import files
from PySubdiv.create_control_cage import control_cage

# First step: import all the necessary meshes from .obj file
original_meshes = [files.read('meshes/gemp_mesh_0_scaled.obj'),
                   files.read('meshes/gemp_mesh_1_cut.obj'), files.read('meshes/gemp_mesh_1_cut_2.obj'),
                   files.read('meshes/gemp_mesh_2_cut_1.obj'), files.read('meshes/gemp_mesh_2_cut_2.obj'),
                   files.read('meshes/gemp_mesh_3_cut_1.obj'), files.read('meshes/gemp_mesh_3_cut_2.obj'),
                   files.read('meshes/gemp_mesh_4_cut_1.obj'), files.read('meshes/gemp_mesh_4_cut_2.obj')]
# Second step load the module control_cage: it will take our list of original meshes as the first parameter, then
# we can pass additional keyword arguments to let the algorithm compute some critical vertex positions for us.
# The function control_cage will return a mesh, so we need to store a reference in a variable:
# find_vertices will find vertices lying on the boundary of the input meshes and on the corner and displays them in
# the viewer. calc_intersection tries to find vertices where two planes intersect, as the input meshes don't have
# an intersection it is here set to False. add_boundary_box will calculate the four corner vertices of the meshes'
# boundary box. This is here necessary to form a watertight mesh in the end. use_dynamic_faces will allow us to set
# the vertices and faces of the control cage as static or dynamic. This constrains will later help in the approximation.
# If the argument simple_subdivision is set to true the final mesh will be simply subdivided. The iteration can be
# controlled by the parameter iteration_subdivision

control_cage = control_cage.create_control_cage(original_meshes, find_vertices=True, calc_intersection=False,
                                                add_boundary_box=True, use_dynamic_faces=True, simple_subdivision=True,
                                                iteration_subdivision=1)


# when the control cage is created we want to save the mesh and the dictionary of the mesh property, so we can restore
# it later
control_cage.visualize_mesh()
control_cage.save_mesh("controlCage/FaultDomainControlCage.obj")
control_cage.save_data("controlCage/FaultDomainControlCageData")
