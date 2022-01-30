from PySubdiv.data import files

# manual fitting of a control cage to input meshes
# first we load our control cage and the data from the previous example

anticlineControlCage = files.read("ControlCage/anticlineControlCage.obj")
anticlineControlCage.load_data("ControlCage/anticlineControlCageData")

# then we load again our original mesh, here is does not matter if it's separated into parts or joined together.
anticlineOriginal = files.read("meshes/anticline_joined.obj")

# to fit the control cage we can call the method .visualize_mesh_interactive() by additionally passing the original mesh
# The method will return the subdivision surface and changes the control cage in-place.
# We can see now the transparent control cage and in the inside we can see the subdivision surface. Clicking the right
# button we can make the original mesh visible.
# To make a good fit we first set the crease values of the vertices/edges lying on the boundary of the input mesh.
# To add one edge to the selection, we must pick the two vertices on the end. We can do that with as many edges we want
# We are going to pick the edges on the boundary, but not on the inside of one plane. After the selection is made
# we can press the "C" key two set the csv here to one. The vertices are set as creases, and we can see that the
# subdivision surface has changed according to the rules. The next step is to find the vertices lying on the corners of
# the original meshes and select the two opposite vertices of the upper and lower surface. Setting the crease values
# of these edges to 1.0 will enable the corner rule of subdivision surfaces. We now should have sharp features o
# the corners of the subdivision surface fitted to the original mesh. The next step is to change the control points, as
# we still have a large void at the maximums. Pressing the left button enables the sphere widget grabbing a sphere
# allow us to move the associated vertex around. As we only have a small number of control points, we only have little
# control on our subdivision surface


anticlineApproximated = anticlineControlCage.visualize_mesh_interactive(2, anticlineOriginal)
anticlineControlCage.visualize_mesh()
anticlineApproximated.visualize_mesh()

# save the control cage and the subdivision surface

anticlineControlCage.save_mesh("ControlCage/anticlineControlCageManualFit.obj")
anticlineControlCage.save_data("ControlCage/anticlineControlCageManualFitData")
anticlineApproximated.save_mesh("ApproximatedMesh/anticlineApproximatedManual.obj")
anticlineApproximated.save_data("ApproximatedMesh/anticlineApproximatedManualData")

# We can refine the control cage before by applying a simple subdivision once ore twice this might help to fit to more
# complex structures.
# Load the unchanged control cage again and apply simple subdivision

anticlineControlCage = files.read("ControlCage/anticlineControlCage.obj")
anticlineControlCage.load_data("ControlCage/anticlineControlCageData")

anticlineControlCageSimpleSubdivision = anticlineControlCage.simple_subdivision(1)
anticlineApproximated = anticlineControlCageSimpleSubdivision.visualize_mesh_interactive(1, anticlineOriginal)
# save the meshes
anticlineControlCageSimpleSubdivision.visualize_mesh()
anticlineApproximated.visualize_mesh()

anticlineControlCage.save_mesh("ControlCage/anticlineControlCageManualFitRefine.obj")
anticlineControlCage.save_data("ControlCage/anticlineControlCageManualFitRefineData")
anticlineApproximated.save_mesh("ApproximatedMesh/anticlineApproximatedManualRefine.obj")
anticlineApproximated.save_data("ApproximatedMesh/anticlineApproximatedManualRefineData")
