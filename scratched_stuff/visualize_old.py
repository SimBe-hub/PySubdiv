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