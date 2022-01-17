from pysubdiv.main.data import files
from pysubdiv.create_control_cage.control_cage import create_control_cage
from pysubdiv.backend import optimization
from pysubdiv.main import main
from pysubdiv.optimization.variational_minimazation import mesh_optimizer
from pysubdiv.backend import visualize
import numpy as np

control_cage = files.read('anticline_cc_appr_6.obj')
#creases = np.load('anticline_CV_6.npy')
#control_cage.set_crease(creases)
#control_cage.subdivide(2).visualize_mesh()

volumes = control_cage.define_volumes()
print(volumes)

np.save('volumes_anticline_cc_appr_6', volumes)

test = np.load('volumes_anticline_cc_appr_6.npy', allow_pickle=True)
print(test)


sdsd

# import our control cage
anticline_control_cage = files.read('../meshes/anticline/anticline_control_cage.obj')
# first import the additional meshes these can be one mesh list/tuple of meshes
anticlines = files.read('../meshes/anticline/anticline_1.obj')

# i'm calling the method with only one additional mesh passed
anticline_control_cage.visualize_mesh_interactive(2, additional_meshes=anticlines)
# the right button allow us to enable/disable the additional mesh
# the left button still activates the movement of the control points
# i changed the way on how to change the crease values a little bit. We can select now more than one edge. The first
# selected vertices is showed as a sphere when we select the second vertex a white line (our selection is drawn)
# we can now select as many edges as we want the same way
# When we press the C key ( on the keyboard) we are prompted to enter the crease values for the selection
# i think here you must edit my code a little bit to make it work with the gui . The prompt is in line 488 in
# the file visualize.py
# edges where we set a crease value are now marked in yellow

# so the same for a list of meshes

anticlines = [files.read('../meshes/anticline/anticline_1.obj'), files.read('../meshes/anticline/anticline_2.obj')]
anticline_control_cage.visualize_mesh_interactive(2, additional_meshes=anticlines)

# if we don't pass additional meshes the right button will disappear
anticline_control_cage.visualize_mesh_interactive(2)

# in the end save the mesh and the data
anticline_control_cage.save_mesh('meshes/anticline/save_me.obj')
anticline_control_cage.save_data('meshes/anticline/save_me_data')







### I think you may need to change the import statements here, i changed the structure in my case a litte bit so have
# look if errors are thrown here.

from pysubdiv.main.data import files
## from pysubdiv.create_control_cage import control_cage   ### should not be important for you
from quad_mesh_simplify import simplify_mesh  ### should not be important for you
from pysubdiv.main import main
from pysubdiv.optimization.variational_minimazation import mesh_optimizer
from pysubdiv.backend import optimization
from pysubdiv.create_control_cage import control_cage
import pyvista as pv
import numpy as np



#  We will start with the function create_control_cage in the file control_cage.py
# so we will load an original mesh here the two antilines: I load them as a list and pass them to the function in
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
                                                add_boundary_box=False, use_dynamic_faces=True)
# So the function still returns a new pysubdiv mesh and new data, so we must be able to save both of them

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






#original_mesh = files.read('meshes/cube.obj')
control_cage = files.read('../meshes/anticline/anticline_control_cage.obj')
control_cage.load_data('meshes/anticline/anticline_control_cage_data')

test_model = control_cage.model()
test_model['label'] = [i for i in control_cage.data['dynamic_vertices']]


p = pv.Plotter()
p.add_mesh(test_model, color='green', use_transparency=False, show_edges=True)
p.add_point_labels(test_model, "label", point_size=20, font_size=36, tolerance=0.01,
                                   show_points=True, name='label_mesh')

p.set_background("royalblue", top="aliceblue")
p.isometric_view_interactive()
p.show_axes()
p.show_grid()
p.show_bounds(all_edges=True)
p.show()

test_2 = control_cage.subdivide(1)
test = control_cage.simple_subdivision(1)

test_model = test.model()

test_model['label'] = [i for i in test_2.data['dynamic_vertices']]
p = pv.Plotter()
p.add_mesh(test_model, color='green', use_transparency=False, show_edges=True)
p.add_point_labels(test_model, "label", point_size=20, font_size=36, tolerance=0.01,
                                   show_points=True, name='label_mesh')

#control_cage = files.read('meshes/cube_control_cage.obj')

#control_cage.save_data('meshes/anticline/anticline_control_cage_data')
#control_cage.save_mesh('meshes/anticline/anticline_control_cage.obj')
p.set_background("royalblue", top="aliceblue")
p.isometric_view_interactive()
p.show_axes()
p.show_grid()
p.show_bounds(all_edges=True)
p.show()





#
optimizer = mesh_optimizer(control_cage, original_mesh, meshes_to_fit=[], use_dynamic_faces=True, use_bounds_p=True)
optimizer.optimize()

control_cage_optimized = optimizer.control_cage
control_cage_optimized.save_mesh('meshes/anticline/anticline_control_cage_opt.obj')
control_cage_optimized.save_data('meshes/anticline/anticline_control_cage_opt_data')
control_cage_optimized.visualize_mesh()
control_cage_optimized.subdivide(2).visualize_mesh()

df

test_model = control_cage.model()
test_model['label'] = [i for i in control_cage.data['dynamic_vertices']]


p = pv.Plotter()
p.add_mesh(test_model, color='green', use_transparency=False, show_edges=True)
p.add_point_labels(test_model, "label", point_size=20, font_size=36, tolerance=0.01,
                                   show_points=True, name='label_mesh')

p.set_background("royalblue", top="aliceblue")
p.isometric_view_interactive()
p.show_axes()
p.show_grid()
p.show_bounds(all_edges=True)
p.show()





test_2 = control_cage.subdivide(2)
test = control_cage.simple_subdivision(2)

test_model = test.model()

test_model['label'] = [i for i in test_2.data['dynamic_vertices']]
p = pv.Plotter()
p.add_mesh(test_model, color='green', use_transparency=False, show_edges=True)
p.add_point_labels(test_model, "label", point_size=20, font_size=36, tolerance=0.01,
                                   show_points=True, name='label_mesh')

p.set_background("royalblue", top="aliceblue")
p.isometric_view_interactive()
p.show_axes()
p.show_grid()
p.show_bounds(all_edges=True)
p.show()


sd
control_cage = files.read('../meshes/anticline/anticline_control_cage_2.obj')
control_cage.load_data('meshes/anticline/anticline_control_cage_data_2')

control_cage.visualize_mesh()

control_cage_optimizer = mesh_optimizer(control_cage, original_mesh, use_dynamic_faces=True, iterations_subdivision=2)
control_cage_optimizer.optimize(number_iteration=5, iterations_swarm=500, nr_particles=10)
fitted_control_cage = control_cage_optimizer.control_cage
fitted_control_cage.save_mesh('meshes/anticline/anticline_control_cage_fitted2.obj')
fitted_control_cage.save_data('meshes/anticline/anticline_control_cage_fitted2')

fitted_control_cage.visualize_mesh()
fitted_control_cage.subdivide(2).visualize_mesh()
sd








ddfdf

# import a mesh. The mesh should be in obj file format for best results
# the imported mesh is our original 'geological' model which we try to find an approximation for

cube_original = files.read('../meshes/cube.obj')
cube_original.visualize_mesh()


# in the next step we need a coarse mesh or control cage on which we will apply the subdivision algorithm. This is
# done for example by calling the create_control_cage_function, in that case i just used mesh decimation, but the
# workflow is now more or less the same

# noinspection PyTupleAssignmentBalance
# this is just the function/library i used for decimation of the mesh, this is not necessary now but should be
# implemented at a later time

#new_positions, new_face = simplify_mesh(np.array(cube_original.vertices), np.array(cube_original.faces,
                                                                                             #dtype=np.uint32), 8)
#cube_coarse_mesh = main.Mesh(vertices=new_positions, faces=new_face)
#cube_coarse_mesh.visualize_mesh()
#cube_coarse_mesh.save_mesh('meshes/cube_coarse_mesh.obj')

# so now we have a control cage which is the first of our inputs to the optimizer
cube_coarse_mesh = files.read('../meshes/cube_coarse_mesh.obj')
# i also wrote a function to save and load the meshes data this should be implemented on any file read/save related to
# the control cage. What i mean when i created the control cage and saved the .obj file it should also save the data
# when i load the control cage for the optimizer the data should also be loaded.
# when i load the an .obj file to the set_crease methods the users should decide on their own to load the
# data file or not
# when there are questions just ask

# the function is called on the object itself we have to input the file path as a string the file ending .pkl will be
# automatically added when it is not passed.
cube_coarse_mesh.save_data('filename')
# and load data to the same mesh:
cube_coarse_mesh.load_data('filename')
# or a completely new  mesh
new_mesh = files.read('../meshes/cube_coarse_mesh.obj')
new_mesh.load_data('filename')
# or to an empty mesh
new_mesh_empty = main.Mesh()
new_mesh_empty.load_data('filename')

new_mesh.visualize_mesh()

# so now to initialize the optimizer:
# first input is our course mesh/control cage we want to be fitted to our original mesh
# i thought about which parameters are important and in contrast to what i said we should also include 'meshes_to_fit'
# this can have multiple input. The easiest one is None meaning that if our original mesh is divided into parts and a
# list of meshes we fitting our coarse mesh to all of them. The second idea is that when only a certain part of our
# original mesh is important, e.g. we two important geologic structures such as anticlines as different meshes and around
# them is also a boundary box meshed (in parts) we have two meshes for the anticlines in our list and six meshes of tje
# boundary box our list of meshes would look like this:
# ['anticline_1', 'anticline_2', 'bb1', bb2', bb3', bb4', bb5', bb6'] so we would then pass meshes_to_fit = (0,1)
# these are the indices of the anticlines in the list. This is not the best implementation but the best i did come up with
# then we have the parameter 'iterations_subdivision' which is the level of subdivison performed during the optimization
# an integer between 1-4 is good, but maybe write a little warning like: high number of iterations will be slow for
# large meshes
# use_dynamic_faces can be set as True or False. This is important when we work with meshes that are divided into parts
# like when we want to model a simple layer cake with three layers and the mesh is only the three layers and not closed
# but we don't want to model a boundary box. To make the control cage watertight we have to make faces on the side of
# the model, but we don't have any mesh to fit the sides to. The way of handling this is to set vertices and faces as
# 'dynamic' during the creation of the control cage where we have the layers and vertices and faces as 'static' when
# we don't have a mesh to fit to so that these faces and vertices are not changed during the optimization
# in my case set to False as i got my cc through decimation and all faces should be fitted.
# variable_edges, variable_vertices should be set to 'automatic' at the moment. But maybe you can already implement them
# and grey them out or sth. with a little message like 'not implemented at the moment' or sth. Theory behind it is that
# the someone can decide for example that a specific vertex/edge should not change during the optimization
# use_bounds_p: True or False. Boundaries on our control vertices. When set to true the vertices are not changed outside
# the bounding box
# a_z=1, lambda_e=1 positive numbers used during the optimization let the user change them to sth between 1-50 I'm myself
# not quite sure what their effect is during the optimization the author of the paper is not going into these and
# just say positive numbers which are not very sensitive during the optization

cube_coarse_mesh_optimizer = mesh_optimizer(cube_coarse_mesh, cube_original,
                                            meshes_to_fit=None, use_dynamic_faces=False, iterations_subdivision=1,
                                            variable_edges='automatic', variable_vertices='automatic', use_bounds_p=False,
                                            a_z=1, lambda_e=1)

# start the optimization:
# number iterations integer how many time the optimization is performed, five is the default,
# epsilon_0 stop criteria for the algorithm
# iterations_swarm: how many times the swarm optimization is performed 500 is good as a default value
# nr_particles: int how many particles are part of the swarm 10 is good as default
# c1, c2, w parameters for the swarm optimization default are good, maybe grey them out with a tickbox to activiate and
# the hint 'experimental'
# when you test the algorithm take a small mesh and set values of iterations low otherwise you will wait ;)

cube_coarse_mesh_optimizer.optimize(number_iteration=5, epsilon_0=1e-5, iterations_swarm=500, nr_particles=10,
                                    c1=0.5, c2=0.3, w=0.9)



