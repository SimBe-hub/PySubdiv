from pysubdiv.main.data import files
import pyvista as pv

# load the meshes from obj
FaultDomainOriginal = files.read("meshes/FaultsJoined.obj")
FaultDomainApproximated = files.read("ApproximatedMesh/FaultDomainApproximatedEdited.obj")
# convert to PyVista.Polydata for visualizing

FaultDomainOriginal_model = FaultDomainOriginal.model()
FaultDomainApproximated_model = FaultDomainApproximated.model()
# initialize a plotter

p = pv.Plotter()
p.add_mesh(FaultDomainOriginal_model, color='green', use_transparency=False, show_edges=True)
p.add_mesh(FaultDomainApproximated_model, color='red', use_transparency=False, show_edges=True)
p.set_background("royalblue", top="aliceblue")
p.isometric_view_interactive()
p.show_axes()
p.show_grid()
p.show_bounds(all_edges=True)
p.show()

