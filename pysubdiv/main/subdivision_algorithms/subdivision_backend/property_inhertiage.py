from pysubdiv.main.data import data_structure
import numpy as  np


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


def pass_down_dynamic_faces_vertices(dynamic_faces, dynamic_vertices, edges, faces_edges_matrix, fitting_method):
    dynamic_edge = []
    for edge_index in range(len(edges)):
        connected_faces = data_structure.get_edge_face_connectivity(faces_edges_matrix, edge_index)
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
