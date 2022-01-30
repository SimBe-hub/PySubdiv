import pyvista.core.composite

from PySubdiv.backend import automatization
import pyvista as pv
import numpy as np
from PySubdiv.backend import calculation
from PySubdiv.backend import optimization
from PySubdiv import PySubdiv


def create_control_cage(mesh, find_vertices=False, calc_intersection=False, add_boundary_box=False,
                        use_dynamic_faces=True, simple_subdivision=False, iteration_subdivision=1,
                        render_style='wireframe', color='green'):
    p = pv.Plotter()  # initialize the PyVista.Plotter
    points_1 = []  # list to store vertices which will used to create the point cloud
    points_2 = []  # list to store vertices which will used to create second point cloud
    added_triangles = set()  # set to hold triangles, to check for duplicates
    invisible_meshes = []
    selected_points = []  # list for storing tris/quads of vertex indices
    line = []  # array to store line mesh for plotting
    labels = []  # array to store vertex index labels for plotting
    points = []  # list to store vertex label points for plotting
    dynamic_face = []  # list to store dynamic faces
    dynamic_vertices_dict = {}  # dictionary to store dynamic vertices
    vertices_fit = []  #
    control_cage_faces = []
    faces = []

    p_cloud_complete = pv.PolyData()
    p_cloud = pv.PolyData()

    # check if one mesh or list of meshes are passed and create Pyvista models
    if isinstance(mesh, list):
        points_for_labels = []
        original_model = []
        for part_mesh in mesh:
            original_model.append(part_mesh.model())
            com = calculation.center_of_mass(part_mesh)
            part_mesh_kdtree = optimization.build_tree(part_mesh)
            points_for_labels.append(part_mesh.vertices[optimization.query_tree(com, part_mesh_kdtree)[0]])
        label_meshes = pv.PolyData(points_for_labels)
        label_meshes['label'] = ['mesh_' + str(i) for i in range(len(mesh))]
        p.add_point_labels(label_meshes, "label", point_size=20, font_size=36, tolerance=0.01,
                           show_points=True, name='label_mesh')
    else:
        original_model = [mesh.model()]
        mesh = [mesh]

    for mesh_part in mesh:
        if not mesh_part.data["mesh_type"] == 'triangular':
            print('At the moment only triangular meshes are supported')
            return

    # If true: Find vertices which are pickable for creating the control cage
    if find_vertices:
        if isinstance(mesh, list):
            boundary_vertices_arr = []
            corner_vertices_arr = []
            for part_mesh in mesh:
                corner_vertices, boundary_vertices = automatization.find_corner_vertices(part_mesh)
                boundary_vertices_arr.append(boundary_vertices)
                corner_vertices_arr.append(corner_vertices)

            boundary_vertices = np.concatenate(boundary_vertices_arr)
            corner_vertices = np.concatenate(corner_vertices_arr)
            if len(boundary_vertices) + len(corner_vertices) == 0:
                print('Is your input mesh watertight?\n'
                      'Boundary and corner vertices can only be found on separated meshes at the moment')
                points_1.append([mesh[0].vertices[0]])
                points_2.append([mesh[0].vertices[0]])
                p_cloud = pv.PolyData(mesh[0].vertices[0])
                p_cloud_complete = pv.PolyData(mesh[0].vertices[0])

            else:
                points_2.append(corner_vertices)
                points_1.extend([corner_vertices, boundary_vertices])

                p_cloud = pv.PolyData(corner_vertices)
                p_cloud_complete = pv.PolyData(corner_vertices) + pv.PolyData(boundary_vertices)

        if calc_intersection:
            if len(mesh) > 1:
                intersection_vertices = []
                compared_meshes = []
                for mesh_1 in mesh:
                    for mesh_2 in mesh:
                        if mesh_1 == mesh_2:
                            continue
                        elif hash(frozenset([mesh_1, mesh_2])) in compared_meshes:
                            continue
                        else:
                            intersection_vertices.append(automatization.find_intersection_points(mesh_1, mesh_2))
                            compared_meshes.append((hash(frozenset([mesh_1, mesh_2]))))
                intersection_vertices = np.concatenate(intersection_vertices)
                p_cloud_complete = pv.PolyData(intersection_vertices + p_cloud_complete)
                p_cloud = pv.PolyData(intersection_vertices + p_cloud)
                points_1.insert(1, intersection_vertices)
                points_2.append(intersection_vertices)

            else:
                print('Input mesh is only one object.\n'
                      'At the moment the intersection can only be found for separated meshes')

        if add_boundary_box:
            boundary_box = automatization.define_boundary_box(mesh)
            points_1.insert(2, boundary_box)
            points_2.append(boundary_box)
            p_cloud = pv.PolyData(boundary_box) + p_cloud
            p_cloud_complete = pv.PolyData(boundary_box) + p_cloud_complete
    else:
        p_cloud = pv.PolyData(mesh[0].vertices[0])
        p_cloud_complete = pv.PolyData(mesh[0].vertices[0])

    original_model_points = []
    for i in range(len(original_model)):
        original_model_points.append(original_model[i].points)

    p_cloud_complete['label'] = np.array([i for i in range(p_cloud_complete.n_points)])
    p_cloud_complete['used_points'] = np.array([0 for i in range(p_cloud_complete.n_points)])
    n_points_p_cloud = p_cloud.n_points
    arr_for_filling = np.ones((p_cloud_complete.n_points - p_cloud.n_points, 3)) * 1000
    p_cloud.points = np.vstack((p_cloud.points, arr_for_filling))

    added_points = np.zeros(p_cloud.n_points)
    added_points[:n_points_p_cloud] = 1

    p_cloud['label'] = np.array([i for i in range(p_cloud.n_points)])
    p_cloud['used_points'] = np.array([0 for i in range(p_cloud.n_points)])
    p_cloud['added_points'] = added_points

    def change_point_colour(point_cloud, idx_vertices):
        for idx in idx_vertices:
            point_cloud['used_points'][idx] = 1

    def update_point_cloud(point_cloud, idx_vertices, pos_vertices):
        for i in range(len(idx_vertices)):
            if idx_vertices[i] >= len(point_cloud['added_points']):
                pass
            elif point_cloud['added_points'][idx_vertices[i]] == 0:
                point_cloud.points[idx_vertices[i]] = pos_vertices[i]
                point_cloud['added_points'][idx_vertices[i]] = 1

    def append_vertex_point_cloud(point_cloud, pos_vertex):
        point_cloud.points = np.vstack((point_cloud.points, pos_vertex))
        point_cloud['used_points'] = np.append(point_cloud['used_points'], 0)
        point_cloud['label'] = np.append(point_cloud['label'], (point_cloud.n_points - 1))

    def add_point_from_picking(point):
        picked_point_hashed = hash(frozenset(point))  # hash value of the picked point
        hashed_points = []  # list to store hash values of the points
        for ind_point in p_cloud_complete.points:  # iterate over the points of the cloud and calculate hash values
            hashed_points.append(hash(frozenset(ind_point)))  # hashing the points and appending

        if picked_point_hashed in hashed_points:  # test if the hash of the picked point is in the hash list of the pc
            # find corresponding idx of the picked face in the input mesh
            idx_point = np.nonzero(np.array(hashed_points) == picked_point_hashed)[0]
            compare_vert_to_picked = (point == p_cloud_complete.points[idx_point])
            if np.all(compare_vert_to_picked):
                update_point_cloud(p_cloud, idx_point, [point])
            else:
                append_vertex_point_cloud(p_cloud_complete, point)
                append_vertex_point_cloud(p_cloud, point)
        else:
            append_vertex_point_cloud(p_cloud_complete, point)
            append_vertex_point_cloud(p_cloud, point)

    def activate_vertex_picking(check):
        if not check:
            p.remove_actor('disableButton')
            p.add_text('enable vertices', position=(50, 10), font_size=10, color='black', name='enableButton')
            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()

            p.add_point_labels(label_meshes, "label", point_size=20, font_size=36, tolerance=0.01,
                               show_points=True, name='label_mesh')
        else:
            p.remove_actor('enableButton')
            p.add_text('disable vertices', position=(50, 10), font_size=10, color='black', name='disableButton')

            p.remove_actor('label_mesh-points')
            p.remove_actor('label_mesh-labels')

            p.remove_actor(text)
            p.add_text('Press middle button to set vertices for the control cage via console input.\n'
                       'Press right button to undo latest face.\n'
                       'Press left button to disable/activate vertex indices in the plot\n'
                       'Press third button to plot all boundary vertices\n'
                       'Press P under the mouse courser to pick additional points, for example inside the surfaces',
                       position='upper_left', name='textbox', color='black')

            p.add_checkbox_button_widget(pop_latest_face, position=(800, 10), color_on='red', color_off='red', size=35)
            p.add_text('Undo last face', position=(840, 10), font_size=10, color='black')

            points.append(p.add_mesh(p_cloud, point_size=20, scalars='used_points',
                                     cmap=['white', 'black'], show_scalar_bar=False, pickable=False))
            labels.append(p.add_point_labels(p_cloud, "label", point_size=20, font_size=36, tolerance=0.01,
                                             show_points=False))

            p.add_checkbox_button_widget(select_vertices, position=(300, 10), size=35, color_on='green',
                                         color_off='green')
            p.add_text('set vertices', position=(345, 10), font_size=10, color='black')
            p.add_checkbox_button_widget(show_boundary_idx, position=(480, 10), size=35, color_on="purple")
            p.add_text('enable boundary vertices', position=(525, 10), font_size=10, color='black', name='boundary')

    def show_boundary_idx(check):
        if 'enableButton' in p.renderer.actors:
            return
        if not check:
            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()
            p.add_text('enable boundary vertices', position=(525, 10), font_size=10, color='black', name='boundary')
            activate_vertex_picking(True)
        else:

            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()
            points.append(p.add_mesh(p_cloud_complete, point_size=20, scalars='used_points',
                                     cmap=['white', 'black'], show_scalar_bar=False, pickable=False))
            labels.append(p.add_point_labels(p_cloud_complete, "label", point_size=20, font_size=36, tolerance=0.01,
                                             show_points=False))
            p.add_text('disable boundary vertices', position=(525, 10), font_size=10, color='black', name='boundary')

    def pop_latest_face(check):
        if check:
            if len(control_cage_faces) == 0:
                print('no faces defined for control cage')
            else:
                print('saved faces: ', control_cage_faces)
                print('deleted face: ', control_cage_faces[-1])
                added_triangles.discard(tuple(sorted(control_cage_faces.pop())))
                for i in range(3):
                    p.remove_actor(line[-1])
                    line.pop()
                p.remove_actor(faces[-1])
                faces.pop()
                dynamic_face.pop()
                vertices_fit.pop()
                selected_points.clear()
                print('saved faces: ', control_cage_faces)
        else:
            pop_latest_face(True)

    def select_vertices(check):
        if check:
            print('choose three control vertices to form a triangular face')
            while True:
                input_0 = input('vertex one: ') or False
                if not input_0:
                    return
                else:
                    try:
                        selected_points.append(int(input_0))
                        break
                    except ValueError:
                        print(f"The input {input_0} cannot be interpreted as an integer, please try again or "
                              f"leave blank to cancel!")
            while True:
                input_1 = input('vertex two: ') or False
                if not input_1:
                    selected_points.clear()
                    return
                else:
                    try:
                        input_1_int = int(input_1)
                        if input_1_int in selected_points:
                            print(f"the input {input_1} already in selected points, please try again or leave blank"
                                  f"to cancel")
                        else:
                            selected_points.append(input_1_int)
                            break
                    except ValueError:
                        print(f"The input {input_1} cannot be interpreted as an integer, please try again or "
                              f"leave blank to cancel!")
            while True:
                input_2 = input('vertex three: ') or False
                if not input_2:
                    selected_points.clear()
                    return
                else:
                    try:
                        input_2_int = int(input_2)
                        if input_2_int in selected_points:
                            print(f"the input {input_2} already in selected points, please try again or leave blank "
                                  f"to cancel")
                        else:
                            selected_points.append(input_2_int)
                            break
                    except ValueError:
                        print(f"The input {input_2} cannot be interpreted as an integer, please try again or "
                              f"leave blank to cancel!")

            vertices = []  # list to store the coordinates of the three selected vertices
            vertices_sorted = tuple(sorted(selected_points))  # temp tuple for sorted vertices for check of dublicates
            if vertices_sorted in added_triangles:
                print(f"Triangle {selected_points} already picked please make different triangle")
                selected_points.clear()
            else:
                added_triangles.add(tuple(sorted(selected_points)))

                for vert_idx in selected_points:  # iterate over the selected vertices and append to list
                    vertices.append(p_cloud_complete.points[vert_idx].tolist())

                if use_dynamic_faces:
                    # set mesh to fit the selected vertices to
                    mesh_id, verts_mesh = automatization.find_mesh_for_fitting(vertices, mesh)
                    saved_actor = p.renderer.actors['mesh_' + str(mesh_id)]
                    p.remove_actor('mesh_' + str(mesh_id))
                    p.add_mesh(original_model[mesh_id], color='red', use_transparency=False, style='surface',
                               show_edges=True,
                               name='visualisation', pickable=False)

                    while True:
                        conf_mesh = input(f'mesh_{mesh_id} selected for fitting. \n'
                                          f'For conformation please press Enter, or input different mesh index.\n'
                                          f'To set triangle as static write "static" or "s",\n'
                                          f'to cancel write "cancel" or "c": ') or 'y'
                        if conf_mesh == 'y':
                            dynamic_face.append(np.array(str(mesh_id)))
                            vertices_fit.append(np.array(verts_mesh))
                            break

                        elif conf_mesh[0].lower() == 'c':
                            p.add_actor(saved_actor, name='mesh_' + str(mesh_id))
                            p.remove_actor('visualisation')
                            added_triangles.discard(tuple(sorted(selected_points)))
                            selected_points.clear()
                            return

                        elif conf_mesh[0].lower() == 's':
                            dynamic_face.append('s')

                            if add_boundary_box:
                                for i in range(len(selected_points)):
                                    vertex = tuple(vertices[i])
                                    mesh_vertices = set()
                                    for vert in mesh[verts_mesh[i]].vertices:
                                        mesh_vertices.add(tuple(vert))
                                    if vertex not in mesh_vertices:
                                        verts_mesh[i] = 's'
                            vertices_fit.append(np.array(verts_mesh))
                            break
                        else:
                            try:
                                corr_mesh = int(conf_mesh)
                                if corr_mesh > len(mesh) - 1:
                                    print(f"the input index {corr_mesh} is out range for indices of the given meshes, "
                                          f"only {len(mesh)} mesh(es) given as input with a maximal "
                                          f"index of {len(mesh) - 1}")
                                else:
                                    dynamic_face.append(np.array(conf_mesh))
                                    vertices_fit.append(np.ones(len(verts_mesh)) * corr_mesh)
                                    break
                            except ValueError:
                                print(f"The input {conf_mesh} cannot be interpreted, please try again")

                    p.add_actor(saved_actor, name='mesh_' + str(mesh_id))
                    p.remove_actor('visualisation')

                print('selected vertices: ', selected_points)

                update_point_cloud(p_cloud, selected_points, vertices)
                change_point_colour(p_cloud, selected_points)
                change_point_colour(p_cloud_complete, selected_points)

                # selected_points_sorted = np.sort(selected_points)
                selected_points_sorted = selected_points
                face_array = [3, 0, 1, 2]
                faces_for_visual = pv.PolyData(p_cloud_complete.points[selected_points], face_array)
                face_actor = p.add_mesh(faces_for_visual, color='green', pickable=False)
                faces.append(face_actor)
                line1 = pv.Line(p_cloud_complete.points[selected_points_sorted[0]],
                                p_cloud_complete.points[selected_points_sorted[1]])
                line2 = pv.Line(p_cloud_complete.points[selected_points_sorted[1]],
                                p_cloud_complete.points[selected_points_sorted[2]])
                line3 = pv.Line(p_cloud_complete.points[selected_points_sorted[2]],
                                p_cloud_complete.points[selected_points_sorted[0]])
                line1_actor = p.add_mesh(line1, color="blue", line_width=5, pickable=False)
                line2_actor = p.add_mesh(line2, color="blue", line_width=5, pickable=False)
                line3_actor = p.add_mesh(line3, color="blue", line_width=5, pickable=False)
                line.append(line1_actor)
                line.append(line2_actor)
                line.append(line3_actor)
                control_cage_faces.append(np.array(selected_points_sorted))
                print('saved faces: ', control_cage_faces)
                selected_points.clear()
        else:
            select_vertices(True)

    def make_mesh_invisible(check):
        if check:
            mesh_nr = int(input('Please enter number of mesh: '))
            if mesh_nr > len(original_model) - 1:
                print(f'Index {mesh_nr} is out of range, try again.')
            else:
                if mesh_nr in invisible_meshes:
                    print(f'mesh_{mesh_nr} already invisible')
                else:
                    p.add_mesh(original_model[mesh_nr], color=color, opacity=0.0, style=render_style, show_edges=True,
                               name=f'mesh_{mesh_nr}', pickable=False)
                    invisible_meshes.append(mesh_nr)
        else:
            make_mesh_invisible(True)

    def make_mesh_visible(check):
        if check:
            if len(invisible_meshes) == 0:
                print('All meshes are visible')
            else:
                for invisible_mesh in invisible_meshes:
                    print(invisible_mesh)
                    p.add_mesh(original_model[invisible_mesh], color=color, use_transparency=False, style=render_style,
                               show_edges=True, name=f'mesh_{invisible_mesh}')
            invisible_meshes.clear()
        else:
            make_mesh_visible(True)

    for model_nr, model in enumerate(original_model):
        name = 'mesh_' + str(model_nr)
        p.add_mesh(model, color=color, use_transparency=False, style=render_style, show_edges=True, name=name)

    p.enable_point_picking(add_point_from_picking, show_message=False)

    p.add_checkbox_button_widget(make_mesh_invisible, position=(10, 130), color_on='yellow',
                                 color_off='yellow', size=35)
    p.add_text('make mesh \n invisible', position=(10, 85), font_size=10, color='black')
    p.add_checkbox_button_widget(make_mesh_visible, position=(10, 230), color_on='orange',
                                 color_off='orange', size=35)

    p.add_text('make mesh \n visible', position=(10, 185), font_size=10, color='black')

    p.add_checkbox_button_widget(activate_vertex_picking, position=(10, 10), size=35)

    p.isometric_view_interactive()
    text = p.add_text('Press button to select control vertices', position='lower_edge', name='textbox')
    p.set_background('royalblue', top='aliceblue')
    p.show()

    vertices_position = []
    control_cage_faces_flattend = np.array(control_cage_faces).flatten()
    control_cage_unique_idx_sorted, indices, inverse = np.unique(control_cage_faces, return_index=True,
                                                                 return_inverse=True)

    control_cage_faces_unique_idx = np.asanyarray([control_cage_faces_flattend[index] for index in sorted(indices)])
    new_idx = []

    for i in range(len(control_cage_faces)):
        for idx in control_cage_faces[i]:
            new_idx.append(np.nonzero(control_cage_faces_unique_idx == idx)[0][0])

    for idx in control_cage_faces_unique_idx:
        vertices_position.append(p_cloud_complete.points[idx].tolist())

    dynamic_face_flattened = np.array(dynamic_face, dtype=str).flatten()
    control_cage_faces = new_idx
    control_cage_faces_reshaped = np.reshape(new_idx, (-1, 3))

    face_index = 0
    for face in control_cage_faces_reshaped:
        vert_index = 0
        for vert in face:
            dynamic_vertices_dict[vert] = vertices_fit[face_index][vert_index]
            vert_index += 1
        face_index += 1
    dynamic_vertices = np.zeros(len(dynamic_vertices_dict), dtype=str)
    for vertex in dynamic_vertices_dict:
        if not isinstance(dynamic_vertices_dict[vertex], list):
            dynamic_vertices[vertex] = str(dynamic_vertices_dict[vertex])
        else:
            unique_indices, counts = np.unique(dynamic_vertices_dict[vertex], return_counts=True)
            if sum(counts) == len(unique_indices):  # all meshes have the same occurrence take the first one
                dynamic_vertices[vertex] = str(dynamic_vertices_dict[vertex][0])
            else:  # take the mesh with maximal occurence
                index_max_occurrence = np.argmax(counts)
                dynamic_vertices[vertex] = str(dynamic_vertices_dict[vertex][index_max_occurrence])

    control_cage = PySubdiv.Mesh(vertices_position, control_cage_faces_reshaped)
    if use_dynamic_faces:
        control_cage.data['fitting_method'] = 'dynamic_faces'
    else:
        control_cage.data['fitting_method'] = None
    control_cage.data['dynamic_faces'] = dynamic_face_flattened
    control_cage.data['dynamic_vertices'] = dynamic_vertices

    if simple_subdivision:
        control_cage = control_cage.simple_subdivision(iteration_subdivision)

    return control_cage


def define_volumes(mesh):
    """
    Define volumes for gmesh software. Volumes are set by faces which form a manifold part of the mesh. Faces which form
    a volume can be picked in the viewer by hovering over with the mouse and pressing  the 'P' key.
    After picking the user is asked to provide the index/number of volume which the picked face should be assigned to by
    console input. An integer should be provided.

    The function will return a nested list with n-number of volumes. Whereas the index of the list is the index/number
    of the volume and the elements in the nested list are the indices of the faces forming the volume.
    Important: If a volume index/number one must remember that the returned list is continuous. E.g. if volumes 0, 1, 3
    are defined the returned list will have the indices 0, 1, 2, aso volume 3 becomes index 2 in the list.

    Parameters
    --------------
    mesh : PySubdiv mesh object
        mesh on which the volumes are defined

    Returns
    --------------
    volumes_list: nested list of n volumes and the indices of the faces forming the volume

    """
    edges_face_incidence = mesh.data['faces_edges_incidence_list']
    faces_edge_incidence = mesh.data['faces_incidence_list']
    faces_faces_incidence = []
    # model = mesh.model()  # converting PySubdiv mesh to PyVista Polydata
    triangles = []  # list to hold triangles of the mesh
    invisible_triangles = []  # list to hold deleted triangles
    model_vertices = mesh.vertices[mesh.faces]  # vertices of the mesh indexed by faces
    triangles_colors = {}  # dictionary to hold the color of one triangle
    colors = ['blue', 'yellow', 'red', 'orange', 'purple', 'cyan', 'pink', 'beige']

    for i in range(len(model_vertices)):  # define each triangle and append
        triangles_colors[i] = 'green'
        triangles.append(pv.PolyData(model_vertices[i], [3, 0, 1, 2]))
        edges_around_face = edges_face_incidence[i]
        faces_around = set()
        for edge in edges_around_face:
            for face in faces_edge_incidence[edge]:
                if face != i:
                    faces_around.add(face)
        faces_faces_incidence.append(faces_around)

    vertex_dict = {}
    faces_dict = {}
    volumes = {}  # volumes for gmesh
    volume_index_used = []  # list for storing the volumes_idx used for undoing the last action
    selected_cells = []  # list to store selection

    index = 0
    for vertex in mesh.vertices:
        vertex_dict[tuple(vertex.tolist())] = index
        index += 1

    index = 0
    for face in mesh.faces:
        faces_dict[tuple(sorted(face.tolist()))] = index
        index += 1

    # callback function to compare the picked face against the faces of the mesh to find the correct index of the face
    # of the mesh

    def make_selection(picked_face):
        if isinstance(picked_face, type(None)):
            return
        else:
            if isinstance(picked_face, pyvista.core.composite.MultiBlock):
                for cell in picked_face:
                    face_idx = find_face_index(cell)
                    if face_idx in selected_cells:
                        selected_cells.remove(face_idx)
                        p.remove_actor(f"tris_selection_{face_idx}")
                    else:
                        p.add_mesh(triangles[face_idx].extract_feature_edges(), color='white', edge_color='white',
                                   line_width=7, pickable=False, name=f"tris_selection_{face_idx}", scalars=None)
                        selected_cells.append(face_idx)
            else:
                face_idx = find_face_index(picked_face)
                if face_idx in selected_cells:
                    selected_cells.remove(face_idx)
                    p.remove_actor(f"tris_selection_{face_idx}")
                else:
                    p.add_mesh(triangles[face_idx].extract_feature_edges(), color='white', edge_color='white',
                               line_width=7, pickable=False, name=f"tris_selection_{face_idx}", scalars=None)
                    selected_cells.append(face_idx)

    def automatic_selection(check):
        if check:
            if len(selected_cells) == 1:
                start_face = selected_cells[0]
                visited_faces = []
                faces_to_visit = {start_face}
                counter = 0
                connected_faces = np.array(list(faces_faces_incidence[start_face]))
                next_face = start_face

                while len(faces_to_visit) > 0:
                    if counter == 0:
                        faces_to_visit = set()

                    continue_with_next_face = False

                    if len(connected_faces) > 3:
                        connected_faces = []
                        connected_edges = edges_face_incidence[next_face]
                        for connected_edge in connected_edges:
                            if len(faces_edge_incidence[connected_edge]) > 2:
                                pass
                            else:
                                for face_index in faces_edge_incidence[connected_edge]:
                                    if face_index != next_face and face_index not in visited_faces:
                                        connected_faces.append(face_index)

                        connected_faces = np.unique(connected_faces)
                    if len(connected_faces) == 0:
                        next_face = list(faces_to_visit)[np.random.randint(len(faces_to_visit))]
                        faces_to_visit.remove(next_face)
                        rest_of_faces = []
                        connected_edges = edges_face_incidence[next_face]
                        for connected_edge in connected_edges:
                            if len(faces_edge_incidence[connected_edge]) > 2:
                                pass
                            else:
                                for face_index in faces_edge_incidence[connected_edge]:
                                    if face_index != next_face and face_index not in visited_faces:
                                        rest_of_faces.append(face_index)
                        continue_with_next_face = True

                    else:
                        next_face = connected_faces[np.random.randint(len(connected_faces))]
                        rest_of_faces = connected_faces[np.nonzero(connected_faces != next_face)[0]]  # 2

                    while not continue_with_next_face:
                        if len(rest_of_faces) == 0:
                            if len(faces_to_visit) == 0:
                                print('Oops, no faces to visit anymore. This can happen when a non-manifold face is \n'
                                      'selected first. Please try again with a manifold face')
                                return
                            next_face = list(faces_to_visit)[np.random.randint(len(faces_to_visit))]
                            rest_of_faces = []
                            connected_edges = edges_face_incidence[next_face]
                            for connected_edge in connected_edges:
                                if len(faces_edge_incidence[connected_edge]) > 2:
                                    pass
                                else:
                                    for face_index in faces_edge_incidence[connected_edge]:
                                        if face_index != next_face and face_index not in visited_faces:
                                            rest_of_faces.append(face_index)
                            faces_to_visit.remove(next_face)
                            continue_with_next_face = True
                        else:
                            if next_face in visited_faces:
                                if isinstance(rest_of_faces, np.int64):
                                    next_face = rest_of_faces
                                else:
                                    next_face = rest_of_faces[np.random.randint(len(rest_of_faces))]  # 1
                                    temp_list = []
                                    for i in rest_of_faces:
                                        if i != next_face and i not in visited_faces:
                                            temp_list.append(i)
                                    rest_of_faces = np.array(temp_list)

                            else:
                                continue_with_next_face = True

                    if isinstance(rest_of_faces, np.int64):
                        if rest_of_faces not in visited_faces:
                            faces_to_visit.add(rest_of_faces)
                    else:
                        for rest_face in rest_of_faces:
                            if rest_face not in visited_faces:
                                faces_to_visit.add(rest_face)

                    visited_faces.append(next_face)
                    if next_face not in selected_cells:
                        p.add_mesh(triangles[next_face].extract_feature_edges(), color='white', edge_color='white',
                                   line_width=7, pickable=False, name=f"tris_selection_{next_face}", scalars=None)
                        selected_cells.append(next_face)
                    connected_faces = np.array(list(faces_faces_incidence[next_face]))
                    counter += 1

            else:
                print('No cell or more than one face selected, please select only one face')
        else:
            automatic_selection(True)

    def clear_selection(check):
        if check:
            for selection in selected_cells:
                p.remove_actor('tris_selection_' + str(selection))
                # p.add_mesh(triangles[selection], color='green', use_transparency=False, show_edges=True,
                # name="tris_" + str(selection))
            selected_cells.clear()
        else:
            clear_selection(True)

    def find_face_index(picked_face_passed):
        picked_face_vertices = picked_face_passed.points
        vertex_indices_from_mesh = []
        for vert in picked_face_vertices:
            vertex_indices_from_mesh.append(vertex_dict[tuple(vert.tolist())])
        face_idx = faces_dict[tuple(sorted(vertex_indices_from_mesh))]
        return face_idx

    def define_volume(face_idx, volume_index):
        if volume_index in volumes:
            if face_idx in volumes[volume_index]:
                print(f"face {face_idx} already part of volume {volume_index}")
            else:
                volumes[volume_index].append(face_idx)
                print(f'face {face_idx} added to volume {volume_index}')
                print(f'volume {volume_index} contain following faces: {volumes[volume_index]}')
        else:
            volumes[volume_index] = [face_idx]
            print(f'face {face_idx} added to volume {volume_index}')
            print(f'volume {volume_index} contain following faces: {volumes[volume_index]}')

        volume_index_used.append(volume_index)
        triangles_colors[face_idx] = colors[volume_index]
        p.add_mesh(triangles[face_idx], use_transparency=False, show_edges=True,
                   name="tris_" + str(face_idx), color=colors[volume_index])

    def undo_last_face(check):  # callback function to undo last face
        if check:
            if not volume_index_used:  # check if list is not empty
                print('No volumes defined')
            else:
                print(volume_index_used)
                print(volumes)
                volume_idx = volume_index_used.pop()  # pop last volume number of the list and store it
                face_idx, volumes[volume_idx] = volumes[volume_idx][-1], volumes[volume_idx][
                                                                         :-1]  # delete last added face
                p.add_mesh(triangles[face_idx], use_transparency=False, show_edges=True, name="tris_" + str(face_idx),
                           color='green')
                print('face ', face_idx, 'deleted from volume ', volume_idx)
                print('volume', volume_idx, 'contains faces', volumes[volume_idx])
        else:
            undo_last_face(True)

    def print_volumes(check):  # callback function to print out the volume dictionary
        if check:
            print('volumes with contained faces \n', volumes)
        else:
            print_volumes(True)

    def delete_face(face_idx):
        invisible_triangles.append(face_idx)
        p.add_mesh(triangles[face_idx], color=triangles_colors[face_idx], use_transparency=True, opacity=0.8,
                   show_edges=True,
                   pickable=False, name="tris_" + str(face_idx))

    def activate_volumes(check):
        if check:
            if len(selected_cells) == 0:
                print('No faces selected')
            else:
                while True:
                    volume_index = input("volume number (integer) or 'c' to cancel: ") or 'c'
                    try:
                        if volume_index[0].lower() == 'c':
                            return
                        else:
                            volume_index = int(volume_index)
                            break
                    except ValueError:
                        print(
                            f'The input {volume_index} cannot be interpreted as an integer, please try again or enter '
                            f'"cancel" to cancel')
                for cell in selected_cells:
                    define_volume(cell, volume_index)
        else:
            activate_volumes(True)

    def make_triangles_visible(check):
        if check:
            if len(invisible_triangles) == 0:
                print('All triangles are visible')
            else:
                for idx_invisible_tris in invisible_triangles:
                    p.add_mesh(triangles[idx_invisible_tris], color=triangles_colors[idx_invisible_tris],
                               use_transparency=False, show_edges=True, name="tris_" + str(idx_invisible_tris))
                invisible_triangles.clear()
        else:
            make_triangles_visible(True)

    def make_invisible(check):
        if check:
            for cell in selected_cells:
                delete_face(cell)
        else:
            make_invisible(True)

    p = pv.Plotter()
    p.set_background('royalblue', top='aliceblue')

    for i in range(len(triangles)):
        p.add_mesh(triangles[i], color='green', use_transparency=False, show_edges=True, name="tris_" + str(i))

    p.enable_cell_picking(callback=make_selection, through=False, show_message=False)
    p.add_text('Press P to select face under the mouse courser.\n'
               'Press R to toggle to rectangle based selection.', color='black')

    p.add_checkbox_button_widget(activate_volumes, value=True, position=(10, 250), size=35,
                                 color_off='red', color_on='red')
    p.add_text('set volumes', position=(60, 253), font_size=14, color='black')

    p.add_checkbox_button_widget(make_invisible, value=True, position=(10, 210), size=35,
                                 color_off='blue', color_on='blue')
    p.add_text('make selection invisible', position=(60, 213), font_size=14, color='black')

    p.add_checkbox_button_widget(clear_selection, position=(10, 170), color_off='green',
                                 color_on='green', size=35)
    p.add_text('clear selection', position=(60, 173), font_size=14, color='black')

    p.add_checkbox_button_widget(automatic_selection, position=(10, 130), color_off='purple',
                                 color_on='purple', size=35)

    p.add_text('automatic selection', position=(60, 133), font_size=14, color='black')

    p.add_checkbox_button_widget(make_triangles_visible, position=(10, 90), color_on='orange',
                                 color_off='orange', size=35)

    p.add_text('make triangles visible', position=(60, 93), font_size=14, color='black')

    p.add_checkbox_button_widget(undo_last_face, position=(10, 50), color_on='lightblue',
                                 color_off='lightblue', size=35)
    p.add_text('undo last face', position=(60, 53), font_size=14, color='black')

    p.add_checkbox_button_widget(print_volumes, position=(10.0, 10.0), color_on='brown', color_off='brown', size=35)
    p.add_text('print volumes', position=(60, 13), font_size=14, color='black')

    p.show()
    sorted_volumes = dict(sorted(volumes.items()))  # sort the dict of volumes
    volumes_list = list(sorted_volumes.values())  # convert dictionary to list, skipped volume number is reduces
    return volumes_list
