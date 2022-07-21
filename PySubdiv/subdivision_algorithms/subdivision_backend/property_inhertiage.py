import numpy as np


def pass_down_volumes(volumes):
    """
    Pass down faces forming volumes to the subdivided mesh. For each old face four new faces are created during
    subdivision. The faces forming a volume must be updated during the subdivision. We iterate over the volumes
    and find the new indices of the old faces in the new faces after subdivision. We now that for each face we have
    four new faces so we can find the first face by multiplying the old index by four and the last face by adding four
    to that value to get the new indices.

    Parameters
    --------------
    volumes: nested list
        nested list of n volumes and the indices of the faces forming the volume

    Returns
    --------------
    volumes_inherited nested: list
        nested list of n volumes and the new indices of the faces forming the volume

    """
    volumes_inherited = []  # list for storing the subdivided faces to the corresponding volume
    for volume in volumes:
        volume_new_faces = []  # list to store the new faces of a volume
        for face in volume:
            first_idx_new_face = face * 4  # first index of subdivided faces used for slicing
            last_idx_new_faces = face * 4 + 4  # last index of subdivided faces used for slicing
            # extending the indices of the four new faces to list
            volume_new_faces.extend(list(range(first_idx_new_face, last_idx_new_faces)))
        volumes_inherited.append(volume_new_faces)
    return volumes_inherited


def pass_down_dynamic_faces_vertices(dynamic_faces, dynamic_vertices, edges, edge_faces_connectivity, fitting_method):
    dynamic_edge = []
    for edge_index in range(len(edges)):
        connected_faces = edge_faces_connectivity[edge_index]
        idx_mesh_for_fitting = dynamic_faces[connected_faces]
        if fitting_method == 'dynamic_faces':
            if len(idx_mesh_for_fitting) > 1:
                if np.all(idx_mesh_for_fitting == 's'):
                    dynamic_edge.append('s')
                else:
                    dynamic_edge.append(idx_mesh_for_fitting[np.nonzero(idx_mesh_for_fitting != 's')[0][0]])
            else:
                dynamic_edge.append(idx_mesh_for_fitting[0])
        else:
            return
    dynamic_vertices = np.concatenate((dynamic_vertices, dynamic_edge))
    dynamic_faces = np.repeat(dynamic_faces, 4)
    return dynamic_faces, dynamic_vertices, dynamic_edge


def pass_down_boundary_vertices(edges_mid, boundary_vertices):
    new_boundary_vertices = np.ones(np.max(edges_mid) + 1) * -1
    new_boundary_vertices[:len(boundary_vertices)] = boundary_vertices

    for vrtx_idx_1, vrtx_idx_2, vrtx_idx_3 in edges_mid:
        edge_boolean_mask = []
        if boundary_vertices[vrtx_idx_1] == 1:
            edge_boolean_mask.append(True)
        else:
            edge_boolean_mask.append(False)
        if boundary_vertices[vrtx_idx_2] == 1:
            edge_boolean_mask.append(True)
        else:
            edge_boolean_mask.append(False)

        if all(edge_boolean_mask):
            new_boundary_vertices[vrtx_idx_3] = 1
        else:
            new_boundary_vertices[vrtx_idx_3] = -1
    return new_boundary_vertices


def pass_down_non_manifold_vertices(edges_mid, non_manifold_vertices):
    new_non_manifold_vertices = np.ones(np.max(edges_mid) + 1) * -1
    new_non_manifold_vertices[:len(non_manifold_vertices)] = non_manifold_vertices

    for vrtx_idx_1, vrtx_idx_2, vrtx_idx_3 in edges_mid:
        edge_boolean_mask = []
        if non_manifold_vertices[vrtx_idx_1] == 1:
            edge_boolean_mask.append(True)
        else:
            edge_boolean_mask.append(False)
        if non_manifold_vertices[vrtx_idx_2] == 1:
            edge_boolean_mask.append(True)
        else:
            edge_boolean_mask.append(False)

        if all(edge_boolean_mask):
            new_non_manifold_vertices[vrtx_idx_3] = 1
        else:
            new_non_manifold_vertices[vrtx_idx_3] = -1
    return new_non_manifold_vertices


def pass_down_layer_boundary_vertices(edges_mid, vertex_property, dynamic_vertices, edge_faces_connectivity,
                                      dynamic_faces):
    new_vertex_property = np.ones(np.max(edges_mid) + 1) * -1
    new_vertex_property[:len(vertex_property)] = vertex_property
    edge_idx = 0
    for vrtx_idx_1, vrtx_idx_2, vrtx_idx_3 in edges_mid:
        edge_boolean_mask = []
        if vertex_property[vrtx_idx_1] >= 0:
            edge_boolean_mask.append(True)
        else:
            edge_boolean_mask.append(False)
        if vertex_property[vrtx_idx_2] >= 0:
            edge_boolean_mask.append(True)
        else:
            edge_boolean_mask.append(False)

        if all(edge_boolean_mask):
            if dynamic_vertices is not None:
                if dynamic_vertices[vrtx_idx_1] == dynamic_vertices[vrtx_idx_2]:
                    if edge_faces_connectivity is not None:
                        connected_faces_set = set()
                        connected_faces = edge_faces_connectivity[edge_idx]

                        for face_idx in connected_faces:
                            connected_faces_set.add(dynamic_faces[face_idx])
                        if len(connected_faces_set) == len(connected_faces):
                            new_vertex_property[vrtx_idx_3] = 1
                        else:
                            new_vertex_property[vrtx_idx_3] = -1
                    else:
                        new_vertex_property[vrtx_idx_3] = 1

                else:
                    new_vertex_property[vrtx_idx_3] = -1
            else:
                new_vertex_property[vrtx_idx_3] = 1
        else:
            new_vertex_property[vrtx_idx_3] = -1
        edge_idx += 1
    return new_vertex_property


def pass_down_boundary_edge_data(old_edges, new_edges, edges_mid, edges_boundary_data):
    idx_boundary_edges = np.nonzero(edges_boundary_data == 1)[0]
    if len(idx_boundary_edges) == 0:
        new_boundary_edges_data = np.zeros(len(new_edges))
        return new_boundary_edges_data
    new_boundary_edges = edges_mid[idx_boundary_edges][:, 2]  # vertex
    max_idx_old_edges = np.max(old_edges)
    indices_new_boundary_edges = set()
    for vertex_idx in new_boundary_edges:
        indices = np.nonzero(new_edges[:, 1] == vertex_idx)[0]
        for idx in indices:
            if new_edges[idx][0] <= max_idx_old_edges:
                indices_new_boundary_edges.add(idx)
    new_boundary_edges_data = np.zeros(len(new_edges))
    new_boundary_edges_data[np.array(list(indices_new_boundary_edges))] = 1
    return new_boundary_edges_data
