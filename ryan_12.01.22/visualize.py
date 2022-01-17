import pyvista as pv
import numpy as np
import pyvista.core.pointset
import calculation
import optimization


def create_model(faces, vertices):
    """
    Creates pyvista.PolyData object, which can be used for plotting.

    Parameters
    ----------
    faces: (n, 4) int
          Indexes of vertices making up the faces
    vertices: (n, 3) float
          Points in space

    Returns
    ---------
    pyvista.PolyData
        pyvista object
    """

    number_points_faces = np.ones(faces.shape[0], dtype=int) * faces.shape[1]
    faces_points = np.hstack(np.column_stack((number_points_faces, faces)))
    return pv.PolyData(vertices, faces_points)


def print_model(pysubdiv_mesh):
    """
    Converts pysubdiv mesh to  pyvista.PolyData object and plots the object

    Parameters
    ----------
    pysubdiv_mesh: PySubdiv mesh obejct

    Returns
    ---------
    pyvista.PolyData
        pyvista object
    """
    model = create_model(pysubdiv_mesh.faces, pysubdiv_mesh.vertices)
    p = pv.Plotter()
    p.add_mesh(model, color='green', use_transparency=False, show_edges=True)
    p.set_background("royalblue", top="aliceblue")
    p.isometric_view_interactive()
    p.show_axes()
    p.show_grid()
    p.show_bounds(all_edges=True)
    factor = p.length * 0.05

    #average_edge_length = pysubdiv_mesh.average_edge_length
    #if average_edge_length <= 1:
    #    factor = 0.2
    #else:
    #    factor = 1

    def recalculate_normals(check):
        if check:
            pysubdiv_mesh.recalculate_face_normals()
            if 'face_normals' in p.renderer.actors:
                show_normals(True)
            if 'vertex_normals' in p.renderer.actors:
                show_vertex_normals(True)
        else:
            recalculate_normals(True)

    def show_normals(check):
        if check:
            arrows = pv.Arrow()
            face_centroids = pysubdiv_mesh.face_centroids
            face_centroids_model = pv.PolyData(face_centroids)
            face_centroids_model['face_normals'] = pysubdiv_mesh.face_normals

            face_normals = face_centroids_model.glyph(orient="face_normals", geom=arrows, factor=factor)
            p.add_mesh(face_normals, show_scalar_bar=False, color='black', name='face_normals')

        else:
            p.remove_actor('face_normals')

    def show_vertex_normals(check):
        if check:
            arrows = pv.Arrow()
            mesh_vertices = pysubdiv_mesh.vertices
            mesh_vertices_poly_data = pv.PolyData(mesh_vertices)
            mesh_vertices_poly_data['vertex_normals'] = pysubdiv_mesh.vertex_normals
            vertex_normals = mesh_vertices_poly_data.glyph(orient="vertex_normals", geom=arrows, factor=factor)
            p.add_mesh(vertex_normals, show_scalar_bar=False, color='black', name='vertex_normals')
        else:
            p.remove_actor('vertex_normals')

    p.add_checkbox_button_widget(show_vertex_normals, position=(10, 260), color_on='lightblue', size=35)
    p.add_text('show vertex normals', position=(60, 263), font_size=14, color='black')
    p.add_checkbox_button_widget(show_normals,  position=(10, 220), color_on='lightblue', size=35)
    p.add_text('show face normals', position=(60, 223), font_size=14, color='black')
    p.add_checkbox_button_widget(recalculate_normals, position=(10, 180), color_on='green', color_off='green',
                                 size=35)
    p.add_text('recalculate normals', position=(60, 183), font_size=14, color='black',
               name='text_recalculate_normals')

    p.show()
    return model


def print_model_interactive(mesh, iteration):
    """
    Creates pyvista.PolyData object and plots the object

    Parameters
    ----------
    mesh: gempy object
    iteration: int
        how many times the algorithm is performed

    Returns
    ---------
    new_verts: (n, 3) float
        position of the new vertices calculated by Catmull-Clark subdivision algorithm

    new_faces: (n, 4) int
        Quadrilaterals which are defined by the indices of the new vertices
    """
    faces = mesh.data['faces']
    vertices = mesh.data['vertices']
    ctr_points = mesh.ctr_points()
    crease_arr = None
    data = []

    def change_sphere_radius(value):
        p.clear_sphere_widgets()
        p.add_sphere_widget(update_surface, center=ctr_points, radius=value, color='red')
        return

    def update_surface(point, i):
        ctr_points[i] = point
        mesh.data['ctr_points'] = ctr_points

        if crease_arr is None:
            sub_div_update = calculation.call_subdiv_algorithm(mesh, iteration, interactive=True)
        else:

            sub_div_update = calculation.call_subdiv_algorithm(mesh, 0, interactive=True)
            for i in range(iteration):
                if i == iteration - 1:
                    sub_div_update.data['creases'] = crease_arr
                    sub_div_update.main()
                else:
                    sub_div_update.main()

        updated_surface = create_model(sub_div_update.data['faces'], sub_div_update.data['vertices'])

        model.points = updated_surface.points
        model.faces = updated_surface.faces
        new_verts.clear()
        new_faces.clear()
        edges.clear()
        crease.clear()
        data.clear()

        new_verts.append(sub_div_update.data['vertices'])
        new_faces.append(sub_div_update.data['faces'])

        edges.append(sub_div_update.edges_list)
        crease.append(sub_div_update.creases_list)
        data.append(sub_div_update.data)

    model = create_model(faces, vertices)
    new_verts = []
    new_faces = []
    edges = []
    crease = []
    selected_points = []
    radius_start = np.mean(abs(vertices[:, 0])) * 0.1
    p = pv.Plotter()
    p.add_slider_widget(change_sphere_radius, [0, radius_start * 2], title='change radius of spheres')
    # p.add_sphere_widget(update_surface, center=ctr_points, radius=radius[-1], color='red')

    edges_arr = np.asanyarray(edges[0][iteration - 1])
    crease_arr = np.asanyarray(crease[0][iteration - 1])

    def activate_edge_picking(check):
        if not check:
            p.clear_slider_widgets()
            p.add_slider_widget(change_sphere_radius, [0, radius_start * 2], title='change radius of spheres')
            # p.add_sphere_widget(update_surface, center=ctr_points, radius=radius[-1], color='red')
        else:

            p.clear_slider_widgets()
            p.clear_sphere_widgets()
            p.enable_point_picking(callback=callback, show_message='Press P to pick two points under the mouse to '
                                                                   'activate the slider,\n'
                                                                   'when the slider is active stop picking more '
                                                                   'points.\n'
                                                                   'The crease of the edge can be changed between '
                                                                   'every second point.\n'
                                                                   'Press button to accept changes and press again to '
                                                                   'change more creases.',
                                   color='pink', point_size=20, font_size=14,
                                   use_mesh=True, show_point=True, tolerance=0.1)

    def callback(mesh_2, pid):

        if np.max(edges_arr) >= pid >= 0:

            selected_points.append(pid)

            if len(selected_points) == 2:
                selected_points_sorted = np.sort(selected_points)

                index_crease = np.nonzero((edges_arr[:, 0] == selected_points_sorted[0]) &
                                          (edges_arr[:, 1] == selected_points_sorted[1]))[0]

                if len(index_crease) == 0:
                    print('Selected points too far away, please chose second point to be only'
                          ' one vertex away from first one')
                    selected_points.clear()
                    return

                def slider(value):

                    crease_arr[index_crease] = np.round(value, 1)
                    sub_div_update = calculation.call_subdiv_algorithm(mesh, 0, interactive=True)

                    for i in range(iteration):
                        if i == iteration - 1:
                            sub_div_update.data['creases'] = crease_arr
                            sub_div_update.main()
                        else:
                            sub_div_update.main()

                    updated_surface = create_model(sub_div_update.data['faces'], sub_div_update.data['vertices'])
                    model.points = updated_surface.points
                    model.faces = updated_surface.faces

                    selected_points.clear()
                    new_verts.clear()
                    new_faces.clear()
                    edges.clear()
                    # edges.clear()
                    crease.clear()
                    data.clear()

                    new_verts.append(sub_div_update.data['vertices'])
                    new_faces.append(sub_div_update.data['faces'])
                    # edges.append(sub_div_update.data['unique_edges'])
                    edges.append(sub_div_update.edges_list)
                    crease.append(sub_div_update.creases_list)
                    data.append(sub_div_update.data)

                p.add_slider_widget(slider, [0, 1], value=crease_arr[index_crease], title='set crease')

        else:
            print('Crease of edge with vertex', pid, 'cannot be changed. Please chose adjacent vertex')

    p.add_checkbox_button_widget(activate_edge_picking)
    p.add_mesh(model, color='green', use_transparency=False, show_edges=True)
    p.isometric_view_interactive()
    p.add_text('Press button to change crease', position='lower_edge', name='textbox')
    p.show_axes()
    p.show_grid()
    p.show_bounds(all_edges=True)
    p.show()

    data = data[0]

    return data


def set_creases_visualize(mesh):
    faces = mesh.data['faces']
    vertices = mesh.data['vertices']

    if 'unique_edges' in mesh.data:
        edges_arr = mesh.data['unique_edges']
    else:
        edges_arr = mesh.edges_unique()
    if 'creases' in mesh.data:
        crease_arr = mesh.data['creases']
    else:
        crease_arr = mesh.set_crease()

    model = create_model(faces, vertices)
    selected_points = []

    def activate_edge_picking(check):
        if not check:
            p.clear_slider_widgets()

        else:
            p.enable_point_picking(callback=callback, show_message='Press P to pick two points under the mouse to '
                                                                   'activate the slider,\n'
                                                                   'when the slider is active stop picking more '
                                                                   'points.\n'
                                                                   'The crease of the edge can be changed '
                                                                   'with the slider.\n'
                                                                   'Press button to accept changes and press again to '
                                                                   'change more creases.',
                                   color='pink', point_size=20, font_size=14,
                                   use_mesh=True, show_point=True, tolerance=0.1)

    def callback(mesh_2, pid):

        if np.max(edges_arr) >= pid >= 0:

            selected_points.append(pid)

            if len(selected_points) == 2:
                selected_points_sorted = np.sort(selected_points)

                index_crease = np.nonzero((edges_arr[:, 0] == selected_points_sorted[0]) &
                                          (edges_arr[:, 1] == selected_points_sorted[1]))[0]

                if len(index_crease) == 0:
                    print('Selected points too far away, please chose second point to be a connected vertex')
                    selected_points.pop()
                    return

                def slider(value):

                    crease_arr[index_crease] = np.round(value, 1)
                    selected_points.clear()

                p.add_slider_widget(slider, [0, 1], value=crease_arr[index_crease], title='set crease')

        else:
            print('Crease of edge with vertex', pid, 'cannot be changed. Please chose adjacent vertex')

    p = pv.Plotter()
    p.add_mesh(model, color='green', use_transparency=False, show_edges=True)
    p.add_checkbox_button_widget(activate_edge_picking)
    p.isometric_view_interactive()
    p.add_text('Press button to change crease', position='lower_edge', name='textbox')
    p.show_axes()
    p.show_grid()
    p.show_bounds(all_edges=True)
    p.show()

    return crease_arr


def geodesic_path(mesh):
    faces = mesh.data['faces']
    vertices = mesh.data['vertices']
    model = create_model(faces, vertices)
    selected_points = []
    geodesic_path_indices = []
    line = []
    labels = []

    def activate_edge_picking(check):
        if not check:
            # p.remove_actor(line)
            p.remove_actor(labels)



        else:
            p.remove_actor(text)
            text_middle = p.add_text(
                'Press middle button to set start and end vertices for the path via console input.\n'
                'Press right button to delete latest path.\n'
                'Press left button to disable/activate vertex indices in th plot', position='upper_left',
                name='textbox')
            p.add_checkbox_button_widget(pop_latest_path, position=(950, 10))

            model['label'] = [i for i in range(model.n_points)]
            labels.append(p.add_point_labels(model, "label", point_size=20, font_size=36, tolerance=0.01))
            p.add_checkbox_button_widget(select_vertices, position=(500, 10))

            # p.enable_point_picking(callback=callback, show_message='Press P to pick two points under the mouse courser.\n'
            #                                                       'The selected points are used to calculate the shortest path\n'
            #                                                       'When the line is drawn two new vertices can be chosen.\n'
            #                                                       'Press button again to save vertices indices on the paths to an array.\n'
            #                                                       'Press again to select additional paths\n'
            #                                                       'Press right bottom to delete latest path',
            #
            #                      color='pink', point_size=20, font_size=14,
            #                      use_mesh=True, show_point=True, tolerance=0.1)

    def callback(mesh, pid):
        if len(vertices) >= pid >= 0:
            selected_points.append(pid)
            if len(selected_points) == 1:
                print('first point selected')
            elif len(selected_points) == 2:
                print('second point selected')
                selected_points_sorted = np.sort(selected_points)
                geo = model.geodesic(selected_points_sorted[0], selected_points_sorted[1])
                geodesic_path_indices.append(geo['vtkOriginalPointIds'])
                print('saved path: ', geodesic_path_indices)
                line.append(p.add_mesh(geo, color="blue", line_width=5))
                selected_points.clear()

    def pop_latest_path(check_1):
        if not check_1:
            pass
        else:
            print('saved path: ', geodesic_path_indices)
            print('deleted path: ', geodesic_path_indices[-1])
            geodesic_path_indices.pop()
            p.remove_actor(line[-1])
            line.pop()
            print('saved path: ', geodesic_path_indices)

    def select_vertices(check_2):
        if not check_2:
            pass
        else:
            selected_points.append(int(input('vertex one:')))
            selected_points.append(int(input('vertex two:')))
            print('selected vertices: ', selected_points)
            selected_points_sorted = np.sort(selected_points)
            geo = model.geodesic(selected_points_sorted[0], selected_points_sorted[1])
            geodesic_path_indices.append(geo['vtkOriginalPointIds'])
            print('saved path: ', geodesic_path_indices)
            line.append(p.add_mesh(geo, color="blue", line_width=5))
            selected_points.clear()

    p = pv.Plotter()
    p.add_mesh(model, color='green', use_transparency=False, show_edges=True)

    p.add_checkbox_button_widget(activate_edge_picking, position=(10, 10))
    p.isometric_view_interactive()
    text = p.add_text('Press button to set path', position='lower_edge', name='textbox')
    p.show()
    return geodesic_path_indices


def visualize_subdivision(mesh, iteration, additional_meshes=None):
    p = pv.Plotter()
    mesh.vertices_edges_incidence(to_matrix=True)
    model_mesh = mesh.model()
    model_subdivided_mesh = mesh.subdivide(iteration).model()
    point_ids = []  # list to hold point indices
    selection = []  # list to hold vertices forming edge
    edges_with_crease = {}  # dict to hold vertices forming edge where a crease is set
    selection_edge_idx = []  # list to hold edge idx
    p.add_mesh(model_mesh, color='green', opacity=0.5, use_transparency=True, show_edges=True, name='control_cage')
    p.add_mesh(model_subdivided_mesh, color='green', use_transparency=False, show_edges=False, pickable=False,
               name='subdivided')

    radius_start = [p.length * 0.025]

    def enable_sphere_widget(check):
        def change_sphere_radius(value):
            p.clear_sphere_widgets()
            p.add_sphere_widget(update_control_cage, center=mesh.ctr_points, radius=value, color='red',
                                test_callback=False)
            radius_start[0] = value

        if check:
            for point in point_ids:
                p.remove_actor(f"point_{point}")
            for edge_idx in selection_edge_idx:
                p.remove_actor(f"selection{edge_idx}")

            selection.clear()
            point_ids.clear()
            selection_edge_idx.clear()
            p.add_slider_widget(change_sphere_radius, [0, radius_start[0]], title='change radius of spheres')
        else:
            p.clear_slider_widgets()
            p.clear_sphere_widgets()

    def update_control_cage(point, idx):
        mesh.vertices[idx] = point
        if idx in edges_with_crease:
            edge_idx = edges_with_crease[idx]
            for idx in edge_idx:
                csv = mesh.creases[idx]
                p.add_lines(model_mesh.points[mesh.edges[idx]], color='yellow',
                            width=(5 * csv) + 0.5, name=f"line{idx}")

        p.add_mesh(model_mesh, color='green', opacity=0.5, use_transparency=True, show_edges=True,
                   name='control_cage')

        p.add_mesh(mesh.subdivide(iteration).model(), color='green', use_transparency=False, show_edges=False,
                   pickable=False,
                   name='subdivided')

    def pick_two_points(pv_model, point_id):
        if isinstance(pv_model, pyvista.core.pointset.PolyData):
            point_ids.append(point_id)
            p.add_mesh(pv.PolyData(pv_model.points[point_id]), render_points_as_spheres=True, pickable=False,
                       point_size=radius_start[0], name=f"point_{point_id}")

        if len(point_ids) == 2:
            point_ids_sorted = np.sort(point_ids)
            idx_edge = np.nonzero((mesh.edges[:, 0] == point_ids_sorted[0]) &
                                  (mesh.edges[:, 1] == point_ids_sorted[1]))[0]

            if len(idx_edge) == 0:
                print('Selected points do not form an edge, please chose second point to be a connected vertex')
                point_ids.pop()
                p.remove_actor(f"point_{point_id}")
            else:
                selection.append(point_ids_sorted)
                selection_edge_idx.append(idx_edge)
                p.add_lines(pv_model.points[point_ids], color='white', width=7,
                            name=f"selection{idx_edge}")
                p.remove_actor([f"point_{point_ids[0]}", f"point_{point_ids[1]}"])
                point_ids.clear()



    if additional_meshes is None:
        offset_for_button = 0.2
        p.add_text(
            "Press P key to pick two vertices forming an edge under the mouse to add to selection\n"
            'Press C key to set crease values for the selection of edges \n'
            'Press button to enable/disable position control of the vertices',
            color='black', font_size=14)
    else:
        if isinstance(additional_meshes, (list, tuple)):
            additional_meshes_polydata = []
            for ind_mesh in additional_meshes:
                additional_meshes_polydata.append(ind_mesh.model())
            additional_mesh = additional_meshes_polydata[0].merge(additional_meshes_polydata[1:])
        else:
            additional_mesh = additional_meshes.model()
        offset_for_button = 0
        p.add_text(
            "Press P key to pick two vertices forming an edge under the mouse to add to selection\n"
            'Press C key to set crease values for the selection of edges \n'
            'Press left button to enable/disable position control of the vertices\n'
            "Press right button to enable/disable additional meshes", color='black', font_size=14)

        def enable_disable_additional_meshes(status):
            if status:
                p.add_mesh(additional_mesh, color='blue', opacity=0.5, use_transparency=True,
                           show_edges=False, pickable=False, name='additional_mesh')
            else:
                p.remove_actor('additional_mesh')

        p.add_checkbox_button_widget(enable_disable_additional_meshes, value=False,
                                     position=(0.7 * p.window_size[0], 10))

    p.enable_point_picking(callback=pick_two_points,
                           show_message=False,
                           color='pink', point_size=20, font_size=14, use_mesh=True,
                           show_point=False, tolerance=0.1)
    p.add_checkbox_button_widget(enable_sphere_widget, color_on='red',
                                 position=((0.3 + offset_for_button) * p.window_size[0], 10))

    def set_crease_values():
        if len(selection_edge_idx) == 0:
            print('No edges selected')
        else:
            print('Set crease sharpness value via console or leave empty to leave current values for the selection')
            # here is the prompt
            csv = float(input('Please enter crease sharpness value: ') or mesh.creases[selection_edge_idx])
            if csv > 1:
                csv = 1.0
            elif csv < 0:
                csv = 0
            mesh.set_crease(np.ones(len(selection_edge_idx)) * csv, np.array(selection_edge_idx).flatten())
            print(f"Crease sharpness value(s) set to: {csv}")
            p.add_mesh(mesh.subdivide(iteration).model(), color='green', use_transparency=False, show_edges=False,
                       pickable=False, name='subdivided')
            for edge_idx in selection_edge_idx:
                p.remove_actor(f"selection{edge_idx}")
                p.add_lines(model_mesh.points[mesh.edges[edge_idx][0]], color='yellow',
                            width=(5 * csv)+0.5, name=f"line{edge_idx[0]}")

            for i in range(len(selection)):
                for vert in selection[i]:
                    edges_with_crease.setdefault(vert, []).append(selection_edge_idx[i][0])
            selection.clear()
            selection_edge_idx.clear()

    p.add_key_event('c', set_crease_values)
    p.set_background("royalblue", top="aliceblue")
    p.isometric_view_interactive()
    p.show_axes()
    p.show_bounds(mesh=model_mesh, grid='front', location='outer', all_edges=True)
    p.show()
    subdivided_mesh = mesh.subdivide(iteration)
    return subdivided_mesh


def visualize_distance(mesh, mesh_to_compare):
    mesh_model = mesh.model()
    distances = optimization.sdf(mesh_to_compare, mesh, return_distance=True)[1]
    mesh_model['distance'] = distances
    additional_mesh = mesh_to_compare.model()
    p = pv.Plotter()
    p.add_mesh(mesh_model, scalars='distance', use_transparency=False, show_edges=False, pickable=False,
                                    name='subdivided')
    p.add_mesh(additional_mesh, color='blue', opacity=0.6, use_transparency=True,
                           show_edges=False, pickable=False, name='additional_mesh')
    p.set_background("royalblue", top="aliceblue")
    p.isometric_view_interactive()
    p.show_axes()
    p.show_grid()
    p.show_bounds(all_edges=True)
    p.show()
