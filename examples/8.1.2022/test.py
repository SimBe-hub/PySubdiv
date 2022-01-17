
### I think you may need to change the import statements here, i changed the structure in my case a litte bit so have
# look if errors are thrown here.

from pysubdiv.main.data import files
## from pysubdiv.create_control_cage import control_cage   ### should not be important for you
from quad_mesh_simplify import simplify_mesh  ### should not be important for you
from pysubdiv.main import main
from pysubdiv.optimization.variational_minimazation import mesh_optimizer
from pysubdiv.backend import optimization
from pysubdiv.create_control_cage import control_cage




#  We will start with the function create_control_cage in the file control_cage.py
# so we will load an original mesh here the two anticlines: I load them as a list and pass them to the function in
# the next step
original_mesh = [files.read('../meshes/anticline/anticline_1.obj'), files.read('../meshes/anticline/anticline_2.obj')]


# the first parameter is the original mesh, this can be either a list of meshes or one individual mesh.
# the second parameter find_vertices is a boolean when True the algorithm tries to calculate vertices on the boundary
# and on the edges of the mesh, then we can pass the parameter calc_intersection (also a boolean) when set to True the
# algorithm tries to find the intersection of two seperate meshes. Will only be performed when find_vertices=True
# the parameter add_boundary_box (bool) will give the corner vertices of the boundary boy when set to True also only
# performed when find_vertices=True- When the parameter use_dynamic_faces (bool) is True now activates the setting for
# the dynamic faces, i got rid of use_mesh_parts (i thinks that was the name of the paramerter) and put both together
# When set to true when a triangular is picked. Will ask the user for the mesh (id) -> automatic
# found mesh is coloured red -> Press Enter to accept -> the input function in Line 293 will then return 'y'
# the second important input is 'cancel' or 'c' (actually any word with 'c' as the first letter will work as it will
# only check the first letter of the input) c will cancel the input and reset the last action,
# the next input is 'static' or just 's' which set the newly created face to static
# the last input can be any other integer < as the number of input meshes (the algorthm will check that) will pass
# the input to the dynamic_faces list
# when an invalid input is given (say any other letter) will repeat the input process

control_cage = control_cage.create_control_cage(original_mesh, find_vertices=True, calc_intersection=False,
                                                add_boundary_box=True, use_dynamic_faces=True)
# So the function still returns a new pysubdiv mesh and new data, so we must be able to save both of them


sdd
control_cage.save_mesh('meshes/save_me.obj')
control_cage.save_data('meshes/save_me_data')

########
# I did make some changes to the optimizer, but just how the optimizer works internally so the parameters
# are more or less just the same
# first change is know that the parameter meshes_to_fit should now be a list (not tuple) anymore. The things i changed
# now works better with list. So meshes_to_fit can be an empty list as seen below, or None. When it is an empty list
# or None it will use every passed original meshes to fit the dynamic faces to. When we pass integers in the list
# e.g. meshes_to_fit=[0,1,2] it will use these to index original_mesh and will only fit the dynamic faces with the
# corresponding indices. I tried to implement some checks to make sure that the indices fit. E.g. when the highest
# dynamic face idx is 3 but only two meshes are passed it will raise errors, but that should not be your concern.
# the next different is that now use_dynamic_faces (bool) will activate or deactivate the fitting with dynamic faces
# so when True will use information on dynamic and static faces and when False will internally set all faces to dynamic
# the rest of the parameters are the same as before

control_cage_optimizer = mesh_optimizer(control_cage, original_mesh, meshes_to_fit=[], use_dynamic_faces=True)
control_cage.optimize()

# get the control cage from the optimizer

cube_control_cage = control_cage_optimizer.control_cage
# save the control cage and data array -> important to extract creases etc.
cube_control_cage.save_mesh('meshes/save_me_mesh.obj')
cube_control_cage.save_data('meshes/save_me_data')



