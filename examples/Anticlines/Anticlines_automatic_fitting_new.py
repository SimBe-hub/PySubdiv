from PySubdiv.data import files
from PySubdiv.optimization import variational_minimazation
import pyvista as pv

# automatic fitting of the control cage to an input mesh with a variational minimization approach after Wu et. al. 2017
# [http://dx.doi.org/10.1007/s41095-017-0088-2]

# import the mesh from obj and the data
anticlineControlCage = files.read("ControlCage/anticlineControlCage.obj")
anticlineControlCage.load_data("ControlCage/anticlineControlCageData")
anticlineControlCage.visualize_mesh()


# create list of input/original meshes
# Caution: Order inside the list should be the same as during the creation of the control cage
original_meshes = [files.read("meshes/anticline_1.obj"), files.read("meshes/anticline_2.obj")]

# initialize the optimizer:
# first argument is the control cage to be optimized. Position of vertices as well as crease sharpness values are
# going to be optimized. Second argument is the list of original meshes.
# With the keyword parameter meshes_to_fit we can constrain the optimization to particular meshes. We can pass a list
# or set it to None to use every mesh. The argument use_dynamic_faces can be passed if we created the control cage with
# dynamic faces and vertices to help the optimizer to fit to the correct mesh. iteration_subdivision will set the level
# of subdivisions performed during the optimization.
# variable_edges and variable_vertices can be used to constrain edges resp. vertices during the optimization. This means
# that the crease sharpness values resp. positions of the control points are not considered/changed -> 'automatic' will
# let the algorithm figure them out. To set bounds for the control points we can set use_bounds_p = True.
# The boundaries for the control points is the bounding box of the control cage.
# a_z and lambda_e coefficients used in the cost functions should be positive.

optimizer = variational_minimazation.MeshOptimizer(anticlineControlCage, original_meshes,
                                    meshes_to_fit=None, use_mesh_data=True,
                                    iterations_subdivision=1, variable_edges='automatic',
                                    a_z=25)

# Perform the optimization:
# Set the number of iterations for the optimization with number_iterations, epsilon_0 is the convergence tolerance.
# The iteration of the swam can be set with iterations_swarm and it's particles with nr_particles.
# The parameters of the swarm can be set with c1 (cognitive parameter), c2 (social parameter), w (inertia parameter)

optimizer.optimize(number_iteration=10, epsilon_0=1e-5, iterations_swarm=100, nr_particles=15, c1=0.5, c2=0.3,
                   w=0.9)

# Get the optimized control cage (position of control points and crease sharpness values)
anticlineControlCageOptimized = optimizer.control_mesh
subdivided_mesh = optimizer.subdivided_mesh
subdivided_mesh.visualize_mesh()


# We can see that the algorithm cannot get a good fit, as it cannot differentiate edges lying on the planes
# from edges on the boundaries. This happens because the vertices on the end of the edges both lies on the boundary and
# the algorithm will constrain these edges the same way as the edges on the boundary. We can circumvent this by refining
# the mesh before.
# anticlineControlCageOptimized.visualize_mesh_interactive(2, original_meshes)


anticlineControlCage = files.read("ControlCage/anticlineControlCage.obj")
anticlineControlCage.load_data("ControlCage/anticlineControlCageData")
anticlineControlCage = anticlineControlCage.simple_subdivision(1)
anticlineControlCage.visualize_mesh()

optimizer = variational_minimazation.MeshOptimizer(anticlineControlCage, original_meshes,
                                    meshes_to_fit=None, use_mesh_data=True,
                                    iterations_subdivision=1, variable_edges='automatic',
                                    a_z=25)
optimizer.optimize(number_iteration=2, epsilon_0=1e-5, iterations_swarm=500, nr_particles=15, c1=0.5, c2=0.3,
                   w=0.9)
anticlineControlCageOptimized = optimizer.control_mesh
anticlineControlCageOptimized.visualize_mesh_interactive(2, original_meshes)

