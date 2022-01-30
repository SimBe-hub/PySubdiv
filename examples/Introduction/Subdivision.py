from PySubdiv.data import files

# import the mesh of the coarse channel
coarseChannel = files.read("Meshes/CoarseChannel.obj")
# first have a look again:
coarseChannel.visualize_mesh()
# perfoming a subivision is easy, we just have to call the method .subdivide(n)
# n is the iteration level of the subdivision surface.
# The default is one time
# subdivide return a new object without altering the input, so we need to assign
# a new object
smoothChannel = coarseChannel.subdivide()
# and visualize
smoothChannel.visualize_mesh()
# we can increase the number to make the mesh even smoother.
# beware that the number of subdivision iterations will increase computational time drastically. Especially when the
# number of vertices is high
#smoothChannel = coarseChannel.subdivide(4)
smoothChannel.visualize_mesh()
# now we have a nice and smooth mesh, but it does not look like a channel bed.
# we can achieve that by setting crease sharpness values to particular edges.
# edges can be accessed by the .edges property

edges_coarseChannel = coarseChannel.edges
for index, edge in enumerate(edges_coarseChannel):
    print(f"Edge index {index} with vertices: {edge}")
# crease sharpness value can be set with the method set_crease([],[]). The method will take two list or numpy arrays.
# The first one is the crease sharpness value between 0.0 and 1.0, whereby 1.0 is sharp and 0 is smooth
# visualizing the mesh helps to find the correct edge indices by finding the correct vertices. The default value is 0.0

coarseChannel.visualize_mesh()

coarseChannel.set_crease([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 5, 39, 13, 37, 20, 18, 2, 36, 26, 17, 29, 38, 41])

smoothChannel = coarseChannel.subdivide(4)
smoothChannel.visualize_mesh()

# This approach might become quite exhausting for large meshes. We can make it easier by calling the method
# .visualize_interactive(n), where n is the number of subdivisions.
# resetting the crease sharpness values to default:

coarseChannel.set_crease([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 5, 39, 13, 37, 20, 18, 2, 36, 26, 17, 29, 38, 41])
# the coarseChannel will be updated in-place and the method returns a new object.
# the interactive viewer will allow us to pick edges by selecting the vertices on the two ends. We can select multiple
# edges and set the crease sharpness value after pressing the "c" key. We can also change the position of the vertices
# after the red button is pressed. We can move around the control vertices of the subdivision surfaces in that way.
smoothChannel = coarseChannel.visualize_mesh_interactive(2)
# print crease sharpness values of the control cage
creases = coarseChannel.creases
for edge, csv in enumerate(creases):
    print(f"Edge {edge} with csv: {csv[0]}")

smoothChannel.visualize_mesh()
# The sharpness of a subdivision surface can be increased by calling the subdivision routine again. We increase
# the level of subdivision by 2
smoothChannel = smoothChannel.subdivide(2)
smoothChannel.visualize_mesh()
# In the end we can save the coarse channel
coarseChannel.save_mesh("Meshes/CoarseChannelEdited.obj")
# If we want to save the crease sharpness data, we can dump the data array of the mesh to pythons .pkl file.
# This data can then later be restored to the object.
coarseChannel.save_data("Meshes/CoarseChannelEditedData")
coarseChannel.load_data("Meshes/CoarseChannelEditedData")
# And we can save the subdivision surface as well.
smoothChannel.save_mesh("Meshes/SmoothChannel.obj")
smoothChannel.save_data("Meshes/SmoothChannelData")
