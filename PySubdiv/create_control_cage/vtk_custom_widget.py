import vtk
import pyvista as pv
from pyvista.utilities import try_callback
import numpy as np
from pyvista.plotting.tools import parse_color
from pyvista import _vtk
import weakref
import logging


class add_position_widget:
    def __init__(self, pyvista_plotter, point, point_indices, mesh, selection_mesh, subdivide_object=None):
        self.pyvista_plotter = pyvista_plotter
        self.point = np.array(point).reshape(-1, 3)
        self.point_pos = np.mean(self.point, axis=0)
        self.old_pos = self.point_pos
        self.point_indices = point_indices
        self.mesh = mesh
        self.selection_mesh = selection_mesh
        print(self.selection_mesh)
        self.subdivide_object = subdivide_object

        self.widget = vtk.vtkHandleWidget()
        self.widget.SetInteractor(pyvista_plotter.iren.interactor)
        handle = vtk.vtkPolygonalHandleRepresentation3D()
        handle.SetTranslationAxis(0)
        handle.SetHandle(pv.Arrow())
        handle.SetWorldPosition(self.point_pos)
        self.widget.SetRepresentation(handle)
        self.widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.move_x)
        self.widget.On()

        self.widget_2 = vtk.vtkHandleWidget()
        self.widget_2.SetInteractor(pyvista_plotter.iren.interactor)
        handle_2 = vtk.vtkPolygonalHandleRepresentation3D()
        handle_2.SetTranslationAxis(1)
        handle_2.SetHandle(pv.Arrow(direction=(0.0, 1.0, 0.0)))
        handle_2.SetWorldPosition(self.point_pos)
        self.widget_2.SetRepresentation(handle_2)
        self.widget_2.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.move_y)
        self.widget_2.On()

        self.widget_3 = vtk.vtkHandleWidget()
        self.widget_3.SetInteractor(pyvista_plotter.iren.interactor)
        handle_3 = vtk.vtkPolygonalHandleRepresentation3D()
        handle_3.SetTranslationAxis(2)
        handle_3.SetHandle(pv.Arrow(direction=(0.0, 0.0, 1.0)))
        handle_3.SetWorldPosition(self.point_pos)
        self.widget_3.SetRepresentation(handle_3)
        self.widget_3.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.move_z)
        self.widget_3.On()

        self.widget_4 = vtk.vtkSphereWidget()
        self.widget_4.SetRepresentationToSurface()
        self.widget_4.SetCenter(self.point_pos)
        self.widget_4.SetInteractor(pyvista_plotter.iren.interactor)
        self.widget_4.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.move_free)
        self.widget_4.On()

    def move_x(self, test, event_id):
        temp = []
        center = test.GetHandleRepresentation().GetWorldPosition()
        self.widget_2.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_3.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_4.SetCenter(center)
        x_value = list(self.widget_4.GetCenter())
        vector = x_value - self.old_pos
        for mesh_idx in self.point_indices.keys():
            for point_idx in self.point_indices[mesh_idx].keys():
                self.mesh[mesh_idx].points[int(point_idx)] += vector
                if self.selection_mesh.n_points == 0:  # when vertices are selected in subdivision state
                    pass
                else:
                    if type(self.point_indices[mesh_idx][point_idx]) == dict:
                        for key in self.point_indices[mesh_idx][point_idx].keys():
                            vert_idx_selection_mesh = key
                        self.selection_mesh.points[vert_idx_selection_mesh] += vector
                    else:
                        self.selection_mesh.points[self.point_indices[mesh_idx][point_idx]] += vector
                temp.append(self.mesh[mesh_idx].points[int(point_idx)])
        temp = np.array(temp).reshape(-1, 3)
        self.old_pos = np.mean(temp, axis=0)
        if self.subdivide_object is not None:
            self.subdivide_object.update_subdiv(self.point_indices[mesh_idx])

    def move_y(self, test, event_id):
        temp = []
        center = test.GetHandleRepresentation().GetWorldPosition()
        self.widget.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_3.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_4.SetCenter(center)
        y_value = list(self.widget_4.GetCenter())
        vector = y_value - self.old_pos
        for mesh_idx in self.point_indices.keys():
            for point_idx in self.point_indices[mesh_idx].keys():
                self.mesh[mesh_idx].points[int(point_idx)] += vector
                if self.selection_mesh.n_points == 0:  # when vertices are selected in subdivision state
                    pass
                else:
                    if type(self.point_indices[mesh_idx][point_idx]) == dict:
                        for key in self.point_indices[mesh_idx][point_idx].keys():
                            vert_idx_selection_mesh = key
                        self.selection_mesh.points[vert_idx_selection_mesh] += vector
                    else:
                        self.selection_mesh.points[self.point_indices[mesh_idx][point_idx]] += vector
                temp.append(self.mesh[mesh_idx].points[int(point_idx)])
        temp = np.array(temp).reshape(-1, 3)
        self.old_pos = np.mean(temp, axis=0)
        if self.subdivide_object is not None:
            self.subdivide_object.update_subdiv(self.point_indices[mesh_idx])

    def move_z(self, test, event_id):
        temp = []
        center = test.GetHandleRepresentation().GetWorldPosition()
        self.widget.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_2.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_4.SetCenter(center)

        z_value = list(self.widget_4.GetCenter())
        vector = z_value - self.old_pos
        for mesh_idx in self.point_indices.keys():
            for point_idx in self.point_indices[mesh_idx].keys():
                self.mesh[mesh_idx].points[int(point_idx)] += vector
                if self.selection_mesh.n_points == 0:  # when vertices are selected in subdivision state
                    pass
                else:
                    if type(self.point_indices[mesh_idx][point_idx]) == dict:
                        for key in self.point_indices[mesh_idx][point_idx].keys():
                            vert_idx_selection_mesh = key
                        self.selection_mesh.points[vert_idx_selection_mesh] += vector
                    else:
                        self.selection_mesh.points[self.point_indices[mesh_idx][point_idx]] += vector
                temp.append(self.mesh[mesh_idx].points[int(point_idx)])
        temp = np.array(temp).reshape(-1, 3)
        self.old_pos = np.mean(temp, axis=0)
        if self.subdivide_object is not None:
            self.subdivide_object.update_subdiv(self.point_indices[mesh_idx])

    def move_free(self, test, event_id):
        temp = []
        center = test.GetCenter()
        self.widget.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_2.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_3.GetHandleRepresentation().SetWorldPosition(center)
        self.widget_4.SetCenter(center)
        all_values = list(self.widget_4.GetCenter())
        vector = all_values - self.old_pos
        for mesh_idx in self.point_indices.keys():
            for point_idx in self.point_indices[mesh_idx].keys():
                self.mesh[mesh_idx].points[int(point_idx)] += vector
                if self.selection_mesh.n_points == 0:  # when vertices are selected in subdivision state
                    pass
                else:
                    if type(self.point_indices[mesh_idx][point_idx]) == dict:
                        for key in self.point_indices[mesh_idx][point_idx].keys():
                            vert_idx_selection_mesh = key
                        self.selection_mesh.points[vert_idx_selection_mesh] += vector
                    else:
                        self.selection_mesh.points[self.point_indices[mesh_idx][point_idx]] += vector
                temp.append(self.mesh[mesh_idx].points[int(point_idx)])
        temp = np.array(temp).reshape(-1, 3)
        self.old_pos = np.mean(temp, axis=0)
        if self.subdivide_object is not None:
            self.subdivide_object.update_subdiv(self.point_indices[mesh_idx])

    def widget_off(self):
        self.widget.Off()
        self.widget_2.Off()
        self.widget_3.Off()
        self.widget_4.Off()


def checkbox_button_widget(pyvista_plotter, callback, idx_mesh, value=False,
                           position=(10., 10.), size=50, border_size=5,
                           color_on='blue', color_off='grey',
                           background_color='white'):
    """Add a checkbox button widget to the scene.

    This is useless without a callback function. You can pass a callable
    function that takes a single argument, the state of this button widget
    and performs a task with that value.

    Parameters
    ----------
    callback : callable
        The method called every time the button is clicked. This should take
        a single parameter: the bool value of the button.

    value : bool, optional
        The default state of the button.

    position : tuple(float), optional
        The absolute coordinates of the bottom left point of the button.

    size : int, optional
        The size of the button in number of pixels.

    border_size : int, optional
        The size of the borders of the button in pixels.

    color_on : str or 3 item list, optional
        The color used when the button is checked. Default is ``'blue'``.

    color_off : str or 3 item list, optional
        The color used when the button is not checked. Default is ``'grey'``.

    background_color : str or sequence, optional
        The background color of the button. Default is ``'white'``.

    Returns
    -------
    vtk.vtkButtonWidget
        The VTK button widget configured as a checkbox button.

    """
    if not hasattr(pyvista_plotter, "button_widgets"):
        pyvista_plotter.button_widgets = []

    def create_button(color1, color2, color3, dims=[size, size, 1]):
        color1 = np.array(parse_color(color1)) * 255
        color2 = np.array(parse_color(color2)) * 255
        color3 = np.array(parse_color(color3)) * 255

        n_points = dims[0] * dims[1]
        button = pv.UniformGrid(dims=dims)
        arr = np.array([color1] * n_points).reshape(dims[0], dims[1], 3)  # fill with color1
        arr[1:dims[0] - 1, 1:dims[1] - 1] = color2  # apply color2
        arr[
        border_size:dims[0] - border_size,
        border_size:dims[1] - border_size
        ] = color3  # apply color3
        button.point_data['texture'] = arr.reshape(n_points, 3).astype(np.uint8)
        return button

    button_on = create_button(color_on, background_color, color_on)
    button_off = create_button(color_on, background_color, color_off)

    bounds = [
        position[0], position[0] + size,
        position[1], position[1] + size,
        0., 0.
    ]

    button_rep = _vtk.vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(2)
    button_rep.SetState(value)
    button_rep.SetButtonTexture(0, button_off)
    button_rep.SetButtonTexture(1, button_on)
    button_rep.SetPlaceFactor(1)
    button_rep.PlaceWidget(bounds)

    button_widget = _vtk.vtkButtonWidget()
    button_widget.SetInteractor(pyvista_plotter.iren.interactor)
    button_widget.SetRepresentation(button_rep)
    button_widget.SetCurrentRenderer(pyvista_plotter.renderer)
    button_widget.On()

    def _the_callback(widget, event):
        state = widget.GetRepresentation().GetState()
        if callable(callback):
            try_callback(callback, bool(state), idx_mesh)

    button_widget.AddObserver(_vtk.vtkCommand.StateChangedEvent, _the_callback)
    pyvista_plotter.button_widgets.append(button_widget)
    return button_widget


def enable_cell_picking(pyvista_plotter, mesh=None, callback=None, through=True,
                        show=True, show_message=True, style='wireframe',
                        line_width=5, color='pink', font_size=18,
                        start=False, **kwargs):
    """Enable picking at cells.

    Press ``"r"`` to enable retangle based selection.  Press
    ``"r"`` again to turn it off. Selection will be saved to
    ``self.picked_cells``. Also press ``"p"`` to pick a single
    cell under the mouse location.

    When using ``through=False``, and multiple meshes are being
    picked, the picked cells in ````self.picked_cells`` will be a
    :class:`MultiBlock` dataset for each mesh's selection.

    Uses last input mesh for input by default.

    .. warning::
       Visible cell picking (``through=False``) will only work if
       the mesh is displayed with a ``'surface'`` representation
       style (the default).

    Parameters
    ----------
    mesh : pyvista.DataSet, optional
        Mesh to select cells from. When ``through`` is ``True``,
        uses last input mesh by default. When ``through`` is
        ``False``, all meshes in the scene are available for
        picking and this argument is ignored. If you would like to
        only pick a single mesh in the scene, use the
        ``pickable=False`` argument when adding the other meshes
        to the scene.

    callback : function, optional
        When input, calls this function after a selection is made.
        The picked_cells are input as the first parameter to this
        function.

    through : bool, optional
        When ``True`` (default) the picker will select all cells
        through the mesh. When ``False``, the picker will select
        only visible cells on the mesh's surface.

    show : bool
        Show the selection interactively.

    show_message : bool or str, optional
        Show the message about how to use the cell picking tool. If this
        is a string, that will be the message shown.

    style : str
        Visualization style of the selection.  One of the
        following: ``style='surface'``, ``style='wireframe'``, or
        ``style='points'``.  Defaults to ``'wireframe'``.

    line_width : float, optional
        Thickness of selected mesh edges. Default 5.

    color : str, optional
        The color of the selected mesh is shown.

    font_size : int, optional
        Sets the font size of the message.

    start : bool, optional
        Automatically start the cell selection tool.

    **kwargs : dict, optional
        All remaining keyword arguments are used to control how
        the selection is interactively displayed.

    """
    if mesh is None:
        if not hasattr(pyvista_plotter, 'mesh') or pyvista_plotter.mesh is None:
            raise AttributeError('Input a mesh into the Plotter class first or '
                                 'or set it in this function')
        mesh = pyvista_plotter.mesh
    self_ = weakref.ref(pyvista_plotter)

    # make sure to consistently use renderer
    renderer_ = weakref.ref(pyvista_plotter.renderer)

    def end_pick_helper(picker, event_id):
        # Merge the selection into a single mesh
        picked = self_().picked_cells
        if isinstance(picked, pv.MultiBlock):
            if picked.n_blocks > 0:
                picked = picked.combine()
            else:
                picked = pv.UnstructuredGrid()
        # Check if valid
        is_valid_selection = picked.n_cells > 0

        if show and is_valid_selection:
            # Select the renderer where the mesh is added.
            active_renderer_index = self_().renderers._active_index
            for index in range(len(pyvista_plotter.renderers)):
                renderer = pyvista_plotter.renderers[index]
                for actor in renderer._actors.values():
                    mapper = actor.GetMapper()
                    if isinstance(mapper, _vtk.vtkDataSetMapper) and mapper.GetInput() == mesh:
                        loc = self_().renderers.index_to_loc(index)
                        self_().subplot(*loc)
                        break

            # Use try in case selection is empty
            self_().add_mesh(picked, name='_cell_picking_selection',
                             style=style, color=color,
                             line_width=line_width, pickable=False,
                             reset_camera=False, **kwargs)

            # Reset to the active renderer.
            loc = self_().renderers.index_to_loc(active_renderer_index)
            self_().subplot(*loc)

            # render here prior to running the callback
            self_().render()
        elif not is_valid_selection:
            pyvista_plotter.remove_actor('_cell_picking_selection')
            self_().picked_cells = None

        if callback is not None:
            try_callback(callback, self_().picked_cells)

        # TODO: Deactivate selection tool
        return

    def through_pick_call_back(picker, event_id):
        extract = _vtk.vtkExtractGeometry()
        mesh.cell_data['orig_extract_id'] = np.arange(mesh.n_cells)
        extract.SetInputData(mesh)
        extract.SetImplicitFunction(picker.GetFrustum())
        extract.Update()
        self_().picked_cells = pv.wrap(extract.GetOutput())
        return end_pick_helper(picker, event_id)

    def visible_pick_call_back(picker, event_id):
        picked = pv.MultiBlock()
        x0, y0, x1, y1 = renderer_().get_pick_position()
        if x0 >= 0:  # initial pick position is (-1, -1, -1, -1)
            selector = _vtk.vtkOpenGLHardwareSelector()
            selector.SetFieldAssociation(_vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
            selector.SetRenderer(renderer_())
            selector.SetArea(x0, y0, x1, y1)
            selection = selector.Select()

            for node in range(selection.GetNumberOfNodes()):
                selection_node = selection.GetNode(node)
                if selection_node is None:
                    # No selection
                    continue
                cids = pv.convert_array(selection_node.GetSelectionList())
                print(cids)
                actor = selection_node.GetProperties().Get(_vtk.vtkSelectionNode.PROP())
                if actor.GetProperty().GetRepresentation() != 2:  # surface
                    logging.warning("Display representations other than `surface` will result in incorrect results.")
                smesh = actor.GetMapper().GetInputAsDataSet()
                smesh = smesh.copy()
                print(smesh)
                smesh["original_cell_ids"] = np.arange(smesh.n_cells)
                print(smesh.faces)
                print(smesh.lines[0])
                print(smesh.n_lines)
                print(smesh.n_cells)
                print(smesh["original_cell_ids"])
                # tri_smesh = smesh.extract_surface().triangulate()
                # print(tri_smesh)
                cids_to_get = smesh.extract_cells(cids)["original_cell_ids"]
                picked.append(smesh.extract_cells(cids_to_get))

            # memory leak issues on vtk==9.0.20210612.dev0
            # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18239#note_973826
            selection.UnRegister(selection)

        if len(picked) == 1:
            self_().picked_cells = picked[0]
        else:
            self_().picked_cells = picked
        return end_pick_helper(picker, event_id)

    area_picker = _vtk.vtkRenderedAreaPicker()
    if through:
        area_picker.AddObserver(_vtk.vtkCommand.EndPickEvent, through_pick_call_back)
    else:
        # NOTE: there can be issues with non-triangulated meshes
        # Reference:
        #     https://github.com/pyvista/pyvista/issues/277
        #     https://github.com/pyvista/pyvista/pull/281
        #     https://discourse.vtk.org/t/visible-cell-selection-hardwareselector-py-example-is-not-working-reliably/1262
        area_picker.AddObserver(_vtk.vtkCommand.EndPickEvent, visible_pick_call_back)

    pyvista_plotter.enable_rubber_band_style()
    pyvista_plotter.iren.set_picker(area_picker)

    # Now add text about cell-selection
    if show_message:
        if show_message is True:
            show_message = "Press R to toggle selection tool"
            if not through:
                show_message += "\nPress P to pick a single cell under the mouse"
        pyvista_plotter.add_text(str(show_message), font_size=font_size, name='_cell_picking_message')

    if start:
        pyvista_plotter.iren._style_class.StartSelect()


def boundary_edges_id(polydata):

    idFilter = vtk.vtkIdFilter()
    idFilter.SetInputData(polydata)
    # idFilter.SetIdsArrayName("ids")
    idFilter.SetPointIds(True)
    idFilter.SetCellIds(False)
    # Available for vtk>=8.3:
    idFilter.SetPointIdsArrayName("ids")
    #idFilter.SetCellIdsArrayName("ids_cell")
    idFilter.Update()

    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(idFilter.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()

    array = edges.GetOutput().GetPointData().GetArray("ids")
    n = edges.GetOutput().GetNumberOfPoints()
    boundaryIds = []
    for i in range(n):
        boundaryIds.append(array.GetValue(i))
    return boundaryIds

def all_edges_id(polydata):
    idFilter = vtk.vtkIdFilter()
    idFilter.SetInputData(polydata)
    # idFilter.SetIdsArrayName("ids")
    idFilter.SetPointIds(True)
    idFilter.SetCellIds(False)
    # Available for vtk>=8.3:
    idFilter.SetPointIdsArrayName("ids")
    # idFilter.SetCellIdsArrayName(arrayName)
    idFilter.Update()

    edges = _vtk.vtkExtractEdges()
    edges.SetInputDataObject(idFilter.GetOutput())
    edges.Update()

    array = edges.GetOutput().GetPointData().GetArray("ids")
    n = edges.GetOutput().GetNumberOfPoints()
    boundaryIds = []
    for i in range(n):
        boundaryIds.append(array.GetValue(i))
    return boundaryIds

