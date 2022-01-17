from pysubdiv.backend import automatization
import pyvista as pv
import numpy as np
from pysubdiv.backend import calculation
from pysubdiv.backend import optimization
from pysubdiv.main import main


def create_control_cage(mesh, find_vertices=False, calc_intersection=False,
                        add_boundary_box=False, use_dynamic_faces=True, use_mesh_parts=False,
                        render_style='wireframe', color='green'):
    p = pv.Plotter()  # initialize the PyVista.Plotter
    points_1 = []  # list to store vertices which will used to create the point cloud
    points_2 = []  # list to store vertices which will used to create second point cloud
    added_triangles = set()  # set to hold triangles, to check for duplicates

    # check if one mesh or list of meshes are passed and create Pyvista models
    if isinstance(mesh, list):
        if use_dynamic_faces and use_mesh_parts:
            print("Mesh parts are used for fitting the control cage. Dynamic faces are deactivated")
            use_dynamic_faces = False
        points_for_labels = []
        original_model = []
        for part_mesh in mesh:
            if not part_mesh.data["mesh_type"] == 'triangular': return
            original_model.append(part_mesh.model())
            com = calculation.center_of_mass(part_mesh)
            part_mesh_kdtree = optimization.build_tree(part_mesh)
            points_for_labels.append(part_mesh.vertices[optimization.query_tree(com, part_mesh_kdtree)[0]])
        label_meshes = pv.PolyData(points_for_labels)
        label_meshes['label'] = ['mesh_' + str(i) for i in range(len(mesh))]
        p.add_point_labels(label_meshes, "label", point_size=20, font_size=36, tolerance=0.01,
                           show_points=True, name='label_mesh')
    else:
        if not mesh.data["mesh_type"] == 'triangular': return
        original_model = [mesh.model()]
        mesh = [mesh]
        if use_mesh_parts:
            print("Mesh is only one object, fitting to parts is deactivated")
            use_mesh_parts = False

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
            points_2.append(corner_vertices)
            points_1.extend([corner_vertices, boundary_vertices])

        if calc_intersection:
            if use_mesh_parts:
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
                points_1.insert(1, intersection_vertices)
                points_2.append(intersection_vertices)

            else:
                print('Input mesh is only one object.\n'
                      ' At the moment the intersection can only be found for separated meshes')

        if add_boundary_box:
            boundary_box = automatization.define_boundary_box(mesh)
            points_1.insert(2, boundary_box)
            points_2.append(boundary_box)
    else:
        points_1 = [mesh[0].vertices[0]]
        points_2 = [mesh[0].vertices[0]]

    original_model_points = []
    for i in range(len(original_model)):
        original_model_points.append(original_model[i].points)

    selected_points = []  # list for storing tris/quads of vertex indices
    line = []  # array to store line mesh for plotting
    labels = []  # array to store vertex index labels for plotting
    points = []  # list to store vertex label points for plotting
    dynamic_face = []  # list to store dynamic faces
    dynamic_vertices_dict = {}  # dictionary to store dynamic vertices
    vertices_fit = []  #
    control_cage_faces = []
    faces = []

    vertices_cloud = np.concatenate(points_1)
    vertices_cloud_2 = np.concatenate(points_2)

    p_cloud_complete = pv.PolyData(vertices_cloud).clean()

    p_cloud_complete['label'] = [i for i in range(p_cloud_complete.n_points)]
    p_cloud_complete['used_points'] = [0 for i in range(p_cloud_complete.n_points)]

    p_cloud = pv.PolyData(vertices_cloud_2).clean()
    n_points_p_cloud = p_cloud.n_points
    arr_for_filling = np.ones((p_cloud_complete.n_points - p_cloud.n_points, 3)) * 1000
    p_cloud.points = np.vstack((p_cloud.points, arr_for_filling))

    added_points = np.zeros(p_cloud.n_points)
    added_points[:n_points_p_cloud] = 1

    p_cloud['label'] = [i for i in range(p_cloud.n_points)]
    p_cloud['used_points'] = [0 for i in range(p_cloud.n_points)]
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
            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()

            if use_mesh_parts:
                p.add_point_labels(label_meshes, "label", point_size=20, font_size=36, tolerance=0.01,
                                   show_points=True, name='label_mesh')
        else:
            if use_mesh_parts:
                p.remove_actor('label_mesh-points')
                p.remove_actor('label_mesh-labels')

            p.remove_actor(text)
            p.add_text('Press middle button to set vertices for the control cage via console input.\n'
                       'Press right button to undo latest face.\n'
                       'Press left button to disable/activate vertex indices in the plot\n'
                       'Press third button to plot all boundary vertices\n'
                       'Press P under the mouse courser to pick additional points, for example inside the surfaces',
                       position='upper_left', name='textbox', color='black')

            p.add_checkbox_button_widget(pop_latest_face, position=(950, 10))
            points.append(p.add_mesh(p_cloud, point_size=20, scalars='used_points',
                                     cmap=['white', 'black'], show_scalar_bar=False, pickable=False))
            labels.append(p.add_point_labels(p_cloud, "label", point_size=20, font_size=36, tolerance=0.01,
                                             show_points=False))

            p.add_checkbox_button_widget(select_vertices, position=(500, 10))
            p.add_checkbox_button_widget(show_boundary_idx, position=(750, 10))

    def show_boundary_idx(check_3):
        if not check_3:
            pass
        else:
            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()
            points.append(p.add_mesh(p_cloud_complete, point_size=20, scalars='used_points',
                                     cmap=['white', 'black'], show_scalar_bar=False, pickable=False))
            labels.append(p.add_point_labels(p_cloud_complete, "label", point_size=20, font_size=36, tolerance=0.01,
                                             show_points=False))

    def pop_latest_face(check_1):
        if not check_1:
            pass
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

    def select_vertices(check_2):
        if not check_2:
            pass
        else:
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
                            print(f"the input{input_1} already in selected points, please try again or leave blank"
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
                            print(f"the input{input_2} already in selected points, please try again or leave blank "
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
                    dynamic = int(input('dynamic face?'))
                    vertices_fit.append(np.ones(len(selected_points)))
                    if dynamic >= 1:
                        dynamic_face.append(np.array(1))

                    else:
                        dynamic_face.append(np.array(0))

                if use_mesh_parts:  # set mesh to fit the selected vertices to
                    mesh_id, verts_mesh = automatization.find_mesh_for_fitting(vertices, mesh)
                    saved_actor = p.renderer.actors['mesh_' + str(mesh_id)]
                    p.remove_actor('mesh_' + str(mesh_id))
                    p.add_mesh(original_model[mesh_id], color='red', use_transparency=False, style='surface',
                               show_edges=True,
                               name='visualisation', pickable=False)

                    while True:
                        conf_mesh = input(f'mesh_{mesh_id} selected for fitting. '
                                          f'For conformation please press Enter, or input different mesh index. '
                                          f'To set triangle as static write "static", '
                                          f'to cancel write "cancel": ') or 'y'
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
                            dynamic_face.append('static')
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

    p.set_background('royalblue', top='aliceblue')

    model_nr = 0
    for model in original_model:
        name = 'mesh_' + str(model_nr)
        p.add_mesh(model, color=color, use_transparency=False, style=render_style, show_edges=True, name=name)
        model_nr += 1

    p.enable_point_picking(add_point_from_picking, show_message=False)

    p.add_checkbox_button_widget(activate_vertex_picking, position=(10, 10))
    p.isometric_view_interactive()
    text = p.add_text('Press button to select control vertices', position='lower_edge', name='textbox')

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

    print(dynamic_vertices)
    control_cage = main.Mesh(vertices_position, control_cage_faces_reshaped)
    if use_dynamic_faces:
        control_cage.data['fitting_method'] = 'dynamic_faces'
    elif use_mesh_parts:
        control_cage.data['fitting_method'] = 'mesh_parts'
    else:
        control_cage.data['fitting_method'] = None
    control_cage.data['dynamic_faces'] = dynamic_face_flattened
    control_cage.data['dynamic_vertices'] = dynamic_vertices

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
    are defined the returned list will have the indices 0, 1, 2, so volume 3 becomes index 2 in the list.

    Parameters
    --------------
    mesh : PySubdiv mesh object
        mesh on which the volumes are defined

    Returns
    --------------
    volumes_list: nested list of n volumes and the indices of the faces forming the volume

    """

    model = mesh.model()  # converting PySubdiv mesh to PyVista Polydata
    hashed_faces = []  # list for storing vertices making the face as hash values to compare against the picked face
    volumes = {}  # volumes for gmesh
    volume_index_used = []  # list for storing the volumes_idx used for undoing the last action

    for face in mesh.vertices[mesh.faces]:  # iterate over the faces of the mesh and calculate hash values
        flatted_face = face.flatten().tolist()  # flatten and converting to list from numpy array
        hashed_faces.append(hash(frozenset(flatted_face)))  # hashing the list and appending

    # callback function to compare the picked face against the faces of the mesh to find the correct index of the face
    # of the mesh
    def find_face_index(picked_face):
        # vertices of the picked face, flattend and conv. to list
        picked_face_vertices = picked_face.points.flatten().tolist()
        # hashed array
        picked_face_hashed = hash(frozenset(picked_face_vertices))
        if picked_face_hashed in hashed_faces:  # test if the hash of the picked face is in the hash list of the mesh
            # find corresponding idx of the picked face in the input mesh
            idx_face_mesh = np.nonzero(np.array(hashed_faces) == picked_face_hashed)[0]
            volume_index = int(input("volume number: (integer) "))  # user index to set the volumes number
            if len(idx_face_mesh) == 1:  # if there is only one face take the save the index to the volumes
                if volume_index in volumes:  # if the volume is stored in the volumes append the face idx to the volume
                    volumes[volume_index] = np.append(volumes[volume_index], idx_face_mesh)  # appending
                    print('face', idx_face_mesh, 'added to volume ', volume_index)  # console feedback
                    print(volumes[volume_index])
                    volume_index_used.append(volume_index)  # appending the number of the volume to list used for undo
                else:
                    # if the volume does not exist create dict key and store the face index
                    volumes[volume_index] = idx_face_mesh
                    print('face ', idx_face_mesh, 'added to volume ', volume_index)
                    print('volume', volume_index, 'contains faces', volumes[volume_index])
                    volume_index_used.append(volume_index)  # appending the number of the volume to list used for undo
            # if there are two faces with the same hash value, we have to check them against
            # the picked face individually
            else:
                picked_vertices = np.sort(picked_face_vertices)  # sort the vertices to check element wise
                for index in idx_face_mesh:  # iterate over the faces with the same hash value
                    # vertices of the mesh flattend and sorted
                    vertices_mesh = np.sort(mesh.vertices[mesh.faces[index]].flatten())
                    # if all vertices are the same the steps are the same as above. Note: could be rewritten to make
                    # code clearer
                    if all(picked_vertices == vertices_mesh):
                        if volume_index in volumes:
                            volumes[volume_index] = np.append(volumes[volume_index], np.array([index]))
                            print('face', index, 'added to volume ', volume_index)
                            print(volumes[volume_index])
                            volume_index_used.append(volume_index)
                        else:
                            volumes[volume_index] = np.array([index])
                            print('face ', index, 'added to volume ', volume_index)
                            print('volume', volume_index, 'contains faces', volumes[volume_index])
                            volume_index_used.append(volume_index)

    def undo_last_face(check):  # callback function to undo last face
        if not volume_index_used:  # check if list is not empty
            print('No volumes defined')
        else:
            print(volume_index_used)
            print(volumes)
            volume_idx = volume_index_used.pop()  # pop last volume number of the list and store it
            face_idx, volumes[volume_idx] = volumes[volume_idx][-1], volumes[volume_idx][:-1]  # delete last added face
            print('face ', face_idx, 'deleted from volume ', volume_idx)
            print('volume', volume_idx, 'contains faces', volumes[volume_idx])

    def print_volumes(check2):  # callback function to print out the volume dictionary
        print('volumes with contained faces \n', volumes)

    p = pv.Plotter()
    p.set_background('royalblue', top='aliceblue')
    p.add_mesh(model, color='green', use_transparency=False, show_edges=True)
    p.enable_cell_picking(callback=find_face_index, through=False)
    p.add_checkbox_button_widget(undo_last_face, position=(850.0, 10.0))
    p.add_checkbox_button_widget(print_volumes, position=(10.0, 10.0))
    p.show()
    sorted_volumes = dict(sorted(volumes.items()))  # sort the dict of volumes
    volumes_list = list(sorted_volumes.values())  # convert dictionary to list, skipped volume number is reduces
    return volumes_list
