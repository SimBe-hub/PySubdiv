from PySubdiv.data import files
from PySubdiv.create_control_cage import control_cage

# load the original mesh exported from gempy and visualize
anticlineOriginal = files.read("meshes/anticline_joined.obj")
anticlineOriginal.visualize_mesh()
# for creating the control cage we will separate the original mesh into two parts. We can create a list and store
# the two objects there.

anticlineOriginalParts = [files.read("meshes/anticline_1.obj"), files.read("meshes/anticline_2.obj")]

# let's create our control cage:
# The function takes our list of original meshes as the first parameter, then
# we can pass additional keyword arguments to let the algorithm compute some critical vertex positions for us.
# The function control_cage will return a mesh, so we need to store a reference in a variable:
# find_vertices will find vertices lying on the boundary of the input meshes and on the corner and displays them in
# the viewer. calc_intersection tries to find vertices where two planes intersect, as the input meshes don't have
# an intersection it is here set to False. add_boundary_box will calculate the four corner vertices of the meshes'
# boundary box. This is here necessary to form a watertight mesh in the end. use_dynamic_faces will allow us to set
# the vertices and faces of the control cage as static or dynamic. This constrains will later help in the approximation.
# If the argument simple_subdivision is set to true the final mesh will be simply subdivided. The iteration can be
# controlled by the parameter iteration_subdivision.

anticlineControlCage = control_cage.create_control_cage(anticlineOriginalParts, find_vertices=True,
                                                        calc_intersection=False, add_boundary_box=False,
                                                        use_dynamic_faces=True, simple_subdivision=False,
                                                        iteration_subdivision=0)

# The first face picked has the vertices [7, 6, 419], 7 and 6 are on the corner of mesh_1, 419 lies on the boundary
# at the maxima of the anticline. When the triangle is set, the algorithm tries to find the related mesh. For the first
# triangle it is correct and the mesh is  coloured red. We can accept by pressing enter in the console prompt.
# We have more options here: Pressing enter to accept, or input a different mesh index. We can pass "static" or "s" to
# set the triangle to static. Or enter "cancel" or "c" to abort and start over.
# We continue by setting the triangles [6, 418, 419], [419, 418, 5], [5, 418, 4]. These triangles are all related
# to mesh_1. When these 4 triangles are set we can continue with the lower surface mesh_0.
# Here we set the triangles [3, 2, 147], [147, 2, 146], [147, 146, 1], [146, 0, 1] all related to mesh_0
# Now we have to close the sides of the mesh to make it watertight. Make sure that the edges are connected seamless, so
# that we don't have any holes in the mesh. It is difficult to repair it later.
# Starting with triangle [0, 4 146], we are yet again prompted to confirm the mesh. This time the algorithm should
# suggest one of the two surfaces, but this in wrong. As we don't have any underlying mesh, we will set this triangle to
# static by passing the character "s". We continue the same way with the faces: [4, 418, 146], [146, 418, 6],
# [146, 6, 2], [2, 6, 3], [3, 6, 7], [3, 7, 147], [7, 419, 147], [147, 419, 5], [147, 5, 1], [1, 5, 0] and finally
# [0, 5, 4]. When we are finished we can just close the window.
# We want save the control and the information about dynamic faces and vertices

anticlineControlCage.visualize_mesh()
anticlineControlCage.save_mesh("ControlCage/anticlineControlCage.obj")
anticlineControlCage.save_data("ControlCage/anticlineControlCageData")





