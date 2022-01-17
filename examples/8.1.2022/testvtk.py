#import vtk
import pyvista as pv
import vtk_custom_widget


def test_callback(mesh, point):
    print(mesh)
    print(point)
    new_center= vtk_custom_widget.add_position_widget(p, point, mesh)
    print(new_center)



p = pv.Plotter()
p.add_mesh(pv.Box())
p.enable_point_picking(test_callback, use_mesh=True)


p.show(auto_close=False)

