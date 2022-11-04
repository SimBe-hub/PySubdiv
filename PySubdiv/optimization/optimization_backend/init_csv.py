import numpy as np
from PySubdiv.backend import calculation
from PySubdiv.data import data_structure
from PySubdiv import PySubdiv_api
from PySubdiv.backend import optimization


def sharp_creases_from_angles(mesh, epsilon=90):
    """
    Initialization of crease values. The initial sharpness is set based on coarse detection of sharp features.
    To detect whether an edge is a candidate for a crease edge, a threshold is used. The threshold is the angle between
    the normals of two adjacent faces containing the edge. If the angle is  90° it is considered to be a crease. i.e
    the crease value is set to 1 otherwise to 0. Non-manifold edges also get a crease value of 1.

    Parameters
    --------------
    mesh : PySubdiv mesh object
        control cage on which we want to find the crease edges
    epsilon: float
        Threshold angle

    Returns
    --------------
    crease_values : n float
        crease values for each edge in the mesh.
    """
    if 'unique_edges' in mesh.data:  # initialize edges in mesh
        pass
    else:
        mesh.edges_unique()
    if 'faces_edges_matrix' in mesh.data:
        edge_face_matrix = mesh.data['faces_edges_matrix']
    else:
        edge_face_matrix = mesh.faces_edges_incidence()

    edge_connected_faces = []  # list to store the faces connected to an edge
    non_manifold_edges = []  # list to store non_manifold_edges
    boundary_edges = []  # list to store boundary edges

    for i in range(len(mesh.data['unique_edges'])):  # find face pairs, faces connected over one edge
        face_pairs = np.nonzero(edge_face_matrix[:, i] == 1)[0].tolist()
        if len(face_pairs) >= 3:  # find non-manifold edges
            non_manifold_edges.append(i)
        elif len(face_pairs) == 1:
            boundary_edges.append(i)
        else:
            edge_connected_faces.append(face_pairs)

    edge_connected_faces = np.array(edge_connected_faces)  # convert list to numpy array
    first_face = mesh.data['faces'][edge_connected_faces[:, 0]]  # extract the first face
    second_face = mesh.data['faces'][edge_connected_faces[:, 1]]  # extract the second face
    # extract points from the faces to calculate

    normal_vector_first_face_unit = calculation.face_normal(mesh.data['vertices'][first_face])
    normal_vector_second_face_unit = calculation.face_normal(mesh.data['vertices'][second_face])

    # calculate angle between normal vectors
    cos = np.zeros(len(normal_vector_first_face_unit))
    for i in range(len(cos)):
        cos[i] = abs((np.dot(normal_vector_first_face_unit[i], normal_vector_second_face_unit[i])))

    threshold = epsilon * (np.pi / 180)
    crease_values = np.where(np.arccos(cos) >= threshold, 1, 0)

    for edge in non_manifold_edges:
        crease_values = np.concatenate((crease_values[:edge], [0], crease_values[edge:]))
    for edge in boundary_edges:
        crease_values = np.concatenate((crease_values[:edge], [1], crease_values[edge:]))
    return np.array(crease_values)


def sharp_creases_from_dynamic_meshes(mesh, dynamic_meshes):
    """
    Initialization of crease values. The initial sharpness is set based on dynamic meshes resp. static meshes which are
    defined during the construction of the control cage. This is useful when to set the boundary box as static for the
    fitting of the control to the original meshes. If an edge is defined as dynamic it is set as smooth (csv = 0) if a
    edge is defined as static it is set as a  crease (csv = 1)

    Parameters
    --------------
    mesh : PySubdiv mesh object
        control cage on which we want to find the crease edges

    dynamic_meshes: [n] int
        list of the indices of dynamic meshes in the list of original mesh parts

    Returns
    --------------
    crease_values : [n] float
        crease values for each edge in the mesh.
    """

    # check if necessary data is already stored in the mesh
    if 'dynamic_faces' in mesh.data:
        pass
    else:
        raise KeyError('dynamic_faces not in data array of mesh')

    if 'faces_edges_matrix' in mesh.data:
        faces_edges_incidence = mesh.data['faces_edges_matrix']
    else:
        faces_edges_incidence = mesh.faces_edges_incidence()

    crease_values = []  # list to store crease values

    for i in range(len(mesh.data['unique_edges'])):  # loop over each unique edge
        # find faces connected to indexed edge
        connected_faces = data_structure.get_edge_face_connectivity(faces_edges_incidence, i)
        # find index of the mesh to fit to -> order is important first index chosen until better way is found
        idx_mesh_for_fitting = mesh.data['dynamic_faces'][connected_faces]

        if np.any(idx_mesh_for_fitting == 's'):
            crease_values.append(1)  # if edge is static set as crease
        else:
            idx_mesh_for_fitting = idx_mesh_for_fitting[np.nonzero(idx_mesh_for_fitting != 's')].astype(int)
            boolean_mask = []
            for idx in idx_mesh_for_fitting:
                if idx in dynamic_meshes:
                    boolean_mask.append(True)
                else:
                    boolean_mask.append(False)
            if all(boolean_mask):
                crease_values.append(0)  # if edge is in dynamic meshes set as smooth
            else:
                crease_values.append(1)  # if edge is not set to dynamic set as crease

        if isinstance(idx_mesh_for_fitting, list):  # test if list, when list append first index
            if idx_mesh_for_fitting[0] in dynamic_meshes:  # if idx is a dynamic mesh set smooth
                crease_values.append(0)
            else:
                crease_values.append(1)  # if idx is not in dynamic mesh set as crease
        else:
            if idx_mesh_for_fitting in dynamic_meshes:
                crease_values.append(0)  # if idx is a dynamic mesh set as smooth
            else:
                crease_values.append(1)  # if idx is not in dynamic mesh set as crease

    return np.array(crease_values, dtype=float)


def sharp_creases_from_dynamic_faces(mesh):
    """
    Initialization of crease values. The initial sharpness is set based on dynamic faces resp. static faces which are
    defined during the construction of the control cage. If an edge is defined as dynamic it is set as smooth (csv = 0)
    if a edge is defined as static it is set as a  crease (csv = 1)

    Parameters
    --------------
    mesh : PySubdiv mesh object
        control cage on which we want to find the crease edges

    Returns
    --------------
    crease_values : [n] float
        crease values for each edge in the mesh.
    """

    # check if necessary data is already stored in the mesh
    if 'dynamic_faces' in mesh.data:
        pass
    else:
        raise KeyError('dynamic_faces not in data array of mesh')

    crease_values = []  # list to store crease values

    if 'faces_edges_matrix' in mesh.data:  # check if necessary data is already stored in the mesh
        faces_edges_incidence = mesh.data['faces_edges_matrix']
    else:
        faces_edges_incidence = mesh.faces_edges_incidence()

    for i in range(len(mesh.data['unique_edges'])):  # loop over each unique edge
        # find faces connected to indexed edge
        connected_faces = data_structure.get_edge_face_connectivity(faces_edges_incidence, i)
        # find index of the mesh to fit to -> order is important first index chosen until better way is found
        face_property = mesh.data['dynamic_faces'][connected_faces].tolist()  # 0 for static and 1 for dynamic face
        if isinstance(face_property, list):  # test if list, when list append first index
            if sum(face_property) > 1:  # if face is dynamic set as smooth
                crease_values.append(0)
            else:
                crease_values.append(1)  # if face is static mesh set as crease
        # else:
        # if face_property == 1:
        # crease_values.append(0)  # if face is a dynamic set as smooth
        # else:
        # crease_values.append(1)  # if face is static set as crease

    return np.array(crease_values)


def sharp_creases_from_boundaries(mesh, original_mesh, surface_to_fit):
    vertex_edge_incidence = []
    faces_edges_incidence = []
    creases = []
    if "vertices_adjacency_list" in mesh.data:
        pass
    else:
        mesh.vertices_connected_to_list()

    if "edge_faces_dictionary" in mesh.data:
        pass
    else:
        mesh.edges_faces_connected()

    if 'boundary_edges_data' in mesh.data:
        boundary_edges_indices = np.nonzero(mesh.data['boundary_edges_data'] == 1)
        creases = np.zeros(len(mesh.edges))
        creases[boundary_edges_indices] = 1
        return creases

    elif 'boundary_vertices' in mesh.data:
        if 'boudary_layer_vertices' in mesh.data:
            index_counter = 0
            for vrtx_idx_1, vrtx_idx_2 in mesh.edges:
                edge_boolean_mask = []
                if len(mesh.data['edge_faces_dictionary'][index_counter]) > 2:
                    creases.append(0)
                else:

                    if mesh.data['boundary_vertices'][vrtx_idx_1] == 1 and \
                            mesh.data['boundary_layer_vertices'][vrtx_idx_1] < 1:
                        edge_boolean_mask.append(True)
                    else:
                        edge_boolean_mask.append(False)
                    if mesh.data['boundary_vertices'][vrtx_idx_2] == 1 \
                            and mesh.data['boundary_layer_vertices'][vrtx_idx_2] == -1:
                        edge_boolean_mask.append(True)
                    else:
                        edge_boolean_mask.append(False)

                    if all(edge_boolean_mask):
                        creases.append(1)
                    else:
                        creases.append(0)
                index_counter += 1
        else:
            index_counter = 0
            for vrtx_idx_1, vrtx_idx_2 in mesh.edges:
                edge_boolean_mask = []
                if len(mesh.data['edge_faces_dictionary'][index_counter]) > 2:
                    creases.append(0)
                else:
                    if mesh.data['boundary_vertices'][vrtx_idx_1] == 1:
                        edge_boolean_mask.append(True)
                    else:
                        edge_boolean_mask.append(False)
                    if mesh.data['boundary_vertices'][vrtx_idx_2] == 1:
                        edge_boolean_mask.append(True)
                    else:
                        edge_boolean_mask.append(False)

                    if all(edge_boolean_mask):
                        creases.append(1)
                    else:
                        creases.append(0)
                index_counter += 1

    else:
        for mesh_part in original_mesh:
            vertex_edge_incidence.append(mesh_part.vertex_edges_dictionary())
            faces_edges_incidence.append(mesh_part.edges_faces_connected())

        for edge in mesh.edges:  # loop over edges of the mesh
            boundary_vertex = []
            for vertex_idx in edge:  # loop over vertices of the edge
                edge_boolean_mask = []  # list to store boundary edges of the original mesh
                temp_mesh = PySubdiv_api.Mesh(vertices=[mesh.vertices[vertex_idx]])
                temp_mesh.data['dynamic_vertices'] = mesh.data['dynamic_vertices'][vertex_idx]
                mesh_id, vert_id = optimization.sdf_with_meshes(
                    np.array(original_mesh)[surface_to_fit].tolist(), temp_mesh)[:2]
                if mesh_id[0] == 's':
                    boundary_vertex.append(False)
                else:
                    verts_edge_con = vertex_edge_incidence[mesh_id[0]][vert_id[0]]
                    for edge_index in verts_edge_con:
                        if len(faces_edges_incidence[mesh_id[0]][edge_index]) > 1:
                            connected_vertices = mesh.data['vertices_adjacency_list'][vertex_idx]
                            if any(mesh.data['dynamic_vertices'][connected_vertices] == 's'):
                                edge_boolean_mask.append(True)
                            else:
                                edge_boolean_mask.append(False)
                        else:
                            edge_boolean_mask.append(True)
                    if any(edge_boolean_mask):
                        boundary_vertex.append(True)
                    else:
                        boundary_vertex.append(False)
            if all(boundary_vertex):
                creases.append(1)
            else:
                creases.append(0)
    return np.array(creases)
