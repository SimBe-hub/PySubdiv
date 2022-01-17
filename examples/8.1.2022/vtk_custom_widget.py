import vtk
import pyvista as pv
from pyvista.utilities import try_callback


def add_position_widget(pyvista_plotter, point, mesh):
    point_pos = mesh.points[point]


    def test(test, event_id):
        center = test.GetHandleRepresentation().GetWorldPosition()
        widget_2.GetHandleRepresentation().SetWorldPosition(center)
        widget_3.GetHandleRepresentation().SetWorldPosition(center)
        widget_4.SetCenter(center)
        mesh.points[point] = list(widget_4.GetCenter())


    def test_2(test, event_id):
        center = test.GetHandleRepresentation().GetWorldPosition()
        widget.GetHandleRepresentation().SetWorldPosition(center)
        widget_3.GetHandleRepresentation().SetWorldPosition(center)
        widget_4.SetCenter(center)
        mesh.points[point] = list(widget_4.GetCenter())

    def test_3(test, event_id):
        center = test.GetHandleRepresentation().GetWorldPosition()
        widget.GetHandleRepresentation().SetWorldPosition(center)
        widget_2.GetHandleRepresentation().SetWorldPosition(center)
        widget_4.SetCenter(center)
        mesh.points[point] = list(widget_4.GetCenter())

    def test_4(test, event_id):
        center = test.GetCenter()
        widget.GetHandleRepresentation().SetWorldPosition(center)
        widget_2.GetHandleRepresentation().SetWorldPosition(center)
        widget_3.GetHandleRepresentation().SetWorldPosition(center)
        widget_4.SetCenter(center)
        mesh.points[point] = list(widget_4.GetCenter())


    widget = vtk.vtkHandleWidget()
    widget.SetInteractor(pyvista_plotter.iren.interactor)
    handle = vtk.vtkPolygonalHandleRepresentation3D()
    handle.SetTranslationAxis(0)
    handle.SetHandle(pv.Arrow())
    handle.SetWorldPosition(point_pos)
    widget.SetRepresentation(handle)
    widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, test)
    widget.On()

    widget_2 = vtk.vtkHandleWidget()
    widget_2.SetInteractor(pyvista_plotter.iren.interactor)
    handle_2 = vtk.vtkPolygonalHandleRepresentation3D()
    handle_2.SetTranslationAxis(1)
    handle_2.SetHandle(pv.Arrow(direction=(0.0, 1.0, 0.0)))
    handle_2.SetWorldPosition(point_pos)
    widget_2.SetRepresentation(handle_2)
    widget_2.AddObserver(vtk.vtkCommand.EndInteractionEvent, test_2)
    widget_2.On()

    widget_3 = vtk.vtkHandleWidget()
    widget_3.SetInteractor(pyvista_plotter.iren.interactor)
    handle_3 = vtk.vtkPolygonalHandleRepresentation3D()
    handle_3.SetTranslationAxis(2)
    handle_3.SetHandle(pv.Arrow(direction=(0.0, 0.0, 1.0)))
    handle_3.SetWorldPosition(point_pos)
    widget_3.SetRepresentation(handle_3)
    widget_3.AddObserver(vtk.vtkCommand.EndInteractionEvent, test_3)
    widget_3.On()

    widget_4 = vtk.vtkSphereWidget()
    widget_4.SetCenter(point_pos)
    widget_4.SetInteractor(pyvista_plotter.iren.interactor)
    widget_4.AddObserver(vtk.vtkCommand.EndInteractionEvent, test_4)
    widget_4.On()