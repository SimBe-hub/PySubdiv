from pysubdiv.main.data import files
## from pysubdiv.create_control_cage import control_cage   ### should not be important for you
from quad_mesh_simplify import simplify_mesh  ### should not be important for you
from pysubdiv.main import main
from pysubdiv.optimization.variational_minimazation import mesh_optimizer
from pysubdiv.backend import optimization
from pysubdiv.create_control_cage import control_cage
import pyvista as pv
import numpy as np
import copy

import sys
from pysubdiv.main.data import data

sys.modules['data'] = data


#test = files.read('meshes/anticline/anticline_cc_with_bb_optimized.obj')
#test.load_data('meshes/anticline/anticline_cc_with_bb_optimized_data')
#creases = np.load('meshes/anticline/anticline_cc_with_bb_optimized_creases.npy')
#test.set_crease(creases)
#test_2 = test.visualize_mesh_interactive(2)
#test_2.visualize_mesh()
#np.save('meshes/anticline/anticline_cc_with_bb_optimized_creases', test.creases)
#test_2.define_volumes()
#sd



# test = files.read('meshes/anticline/anticline_cc_with_bb.obj')
# test.load_data('meshes/anticline/anticline_cc_with_bb_with_volumes')
#
#
#
# print(len(test.faces))
# print(test.faces[198])
# print(type(test.faces[198]))
# test.faces[198] = np.array([101, 81, 79])
# test.faces[189] = np.array([77, 79, 9])
#
# faces = copy.copy(test.faces)
# faces_1 = np.append(faces, np.array([[101, 79, 77], [79, 81, 9]]), axis=0)
#
#
# test.faces = faces_1
# print(test.faces[271])
# print(test.faces[272])
# print(test.data['volumes'][2])
# test.data['volumes'][2].append(271)
# test.data['volumes'][1].append(272)
# test.visualize_mesh()
#
# test.data['dynamic_faces'] = np.append(test.data['dynamic_faces'], 3)
# test.data['dynamic_faces'] = np.append(test.data['dynamic_faces'], 2)
# fac = test.data['dynamic_faces']
# vol = test.data['volumes']
# dyn_verts = test.data['dynamic_vertices']
#
# test = main.Mesh(test.vertices, test.faces)
# test.data['dynamic_vertices'] = dyn_verts
# test.data['volumes'] = vol
# test.data['dynamic_faces'] = fac
#
# test.save_mesh('meshes/anticline/anticline_cc_with_bb_2.obj')
# test.save_data('meshes/anticline/anticline_cc_with_bb_2_data')
#
# df
#
# #test.load_data('meshes/anticline/anticline_cc_with_bb_optimized_data')
# test_model = test.model()
# #test_model['label'] = [i for i in control_cage.data['dynamic_vertices']]
# test_model['label'] = [i for i in range(test_model.n_points)]
#
# p = pv.Plotter()
# p.add_mesh(test_model, color='green', use_transparency=False, show_edges=True)
# p.add_point_labels(test_model, "label", point_size=20, font_size=36, tolerance=0.01,
#                                    show_points=True, name='label_mesh')
#
# p.set_background("royalblue", top="aliceblue")
# p.isometric_view_interactive()
# p.show_axes()
# p.show_grid()
# p.show_bounds(all_edges=True)
# p.show()
#
#
#
#
#
# test.define_volumes()
# test.subdivide().visualize_mesh()
#
#
#
# sdsd

# anticline_original = [files.read('meshes/anticline/anticline_1.obj'), files.read('meshes/anticline/anticline_2.obj'),
#                       files.read('meshes/anticline/anticline_bb_L_1.obj'),
#                       files.read('meshes/anticline/anticline_bb_L_2.obj'),
#                       files.read('meshes/anticline/anticline_bb_L_3.obj'),
#                       files.read('meshes/anticline/anticline_bb_L_4.obj'),
#                       files.read('meshes/anticline/anticline_bb_L_5.obj'),
#                       files.read('meshes/anticline/anticline_bb_L_6.obj')]

#anticline_control_cage = control_cage.create_control_cage(anticline_original,
                                                          #find_vertices=True, use_dynamic_faces=True)
#anticline_control_cage.save_mesh('meshes/anticline/anticline_cc_with_bb.obj')
#anticline_control_cage.save_data('meshes/anticline/anticline_cc_with_bb')

#anticline_control_cage = files.read('meshes/anticline/anticline_cc_with_bb.obj')
#anticline_control_cage.load_data('meshes/anticline/anticline_cc_with_bb_with_volumes')
# anticline_control_cage = files.read('meshes/anticline/anticline_cc_with_bb_2.obj')
# anticline_control_cage.load_data('meshes/anticline/anticline_cc_with_bb_2_data')
# anticline_control_cage.data['fitting_method'] = 'dynamic_faces'
# anticline_control_cage.save_data('meshes/anticline/anticline_cc_with_bb_2_data')
#
# test = anticline_control_cage.subdivide()
# print(len(test.vertices))
# print(len(test.data['dynamic_vertices']))
#
# print(len(anticline_control_cage.vertices))
# print(len(anticline_control_cage.data['dynamic_vertices']))
#
#
# test_model = anticline_control_cage.model()
# #test_model['label'] = [i for i in control_cage.data['dynamic_vertices']]
# test_model['label'] = [i for i in anticline_control_cage.data['dynamic_vertices']]
#
# p = pv.Plotter()
# p.add_mesh(test_model, color='green', use_transparency=False, show_edges=True)
# p.add_point_labels(test_model, "label", point_size=20, font_size=36, tolerance=0.01,
#                                    show_points=True, name='label_mesh')
#
# p.set_background("royalblue", top="aliceblue")
# p.isometric_view_interactive()
# p.show_axes()
# p.show_grid()
# p.show_bounds(all_edges=True)
# p.show()


anticline_control_cage = files.read('../meshes/save_me.obj')
anticline_control_cage = control_cage.create_control_cage(anticline_control_cage)
anticline_control_cage.visualize_mesh()
anticline_control_cage.load_data('meshes/save_me_data')
anticline_control_cage = anticline_control_cage.simple_subdivision(1)
anticline_control_cage.ctr_points = None
print(anticline_control_cage.ctr_points)



test_model = anticline_control_cage.model()
#test_model['label'] = [i for i in control_cage.data['dynamic_vertices']]
test_model['label'] = [i for i in anticline_control_cage.data['dynamic_vertices']]

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


#anticline_control_cage.visualize_mesh()
#print(anticline_control_cage.data['dynamic_faces'])
#print(anticline_control_cage.data['dynamic_vertices'])
print(len(anticline_control_cage.vertices))
anticline_original = [files.read('../meshes/anticline/anticline_1.obj'), files.read(
    '../meshes/anticline/anticline_2.obj')]
#anticline_control_cage.visualize_mesh_interactive(1, anticline_original)

optimizer = mesh_optimizer(anticline_control_cage, anticline_original, meshes_to_fit=[], use_dynamic_faces=True,
                           iterations_subdivision=1, a_z=25)
optimizer.optimize(1, nr_particles=5, iterations_swarm=2)

control_cage_optimized = optimizer.control_cage
control_cage_optimized.visualize_mesh()
control_cage_optimized.visualize_mesh_interactive(2, anticline_original)
control_cage_optimized.subdivide(2).visualize_mesh()
#control_cage_optimized.save_mesh('meshes/anticline/anticline_cc_with_bb_optimized.obj')
#control_cage_optimized.save_data('meshes/anticline/anticline_cc_with_bb_optimized_data')
