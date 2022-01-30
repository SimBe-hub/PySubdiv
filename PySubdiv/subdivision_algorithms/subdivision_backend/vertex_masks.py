import numpy as np


def calculate_smooth_mask_for_vertex(number_faces, number_edges):
    # weights for the regular case with 6 adjacent faces:
    # 1/16 per edge vert and 5/8 for the vertex itself

    vertex_weight = 0.625
    edge_weight = np.ones(number_edges) * 0.0625

    # weights for the unregular case
    if number_faces != 6:
        inv_valence = 1 / number_faces
        cos_theta = np.cos((2 * np.pi) * inv_valence)

        beta = inv_valence * (0.625 - (0.375 + 0.25 * cos_theta) ** 2)
        edge_weight = np.ones(number_edges) * beta
        vertex_weight = 1 - (beta * number_edges)
    return vertex_weight, edge_weight


def calculate_crease_mask_for_vertex(mesh, connected_edge_mids, connected_verts, crease_edges):
    vertex_weight = 0.75
    connected_crease_edges = connected_edge_mids[
        np.isin(connected_edge_mids, crease_edges)]
    connected_crease_verts = np.unique(mesh.data['unique_edges'][connected_crease_edges])
    edge_weight = np.where(np.isin(connected_verts, connected_crease_verts), 0.125, 0)

    return vertex_weight, edge_weight


def fractional_weight(mesh, index_incident_edges):
    transition_sum = np.sum(mesh.data['creases'][index_incident_edges])
    transition_count = np.sum(mesh.data['creases'][index_incident_edges] > 0)
    fw = transition_sum / transition_count  # frictional weight
    return fw


def combine_vertex_masks(vertex_weight_sharp, vertex_weight_smooth, edge_weight_sharp,
                         edge_weight_smooth, pWeight, cWeight):
    vertex_weight_combined = pWeight * vertex_weight_sharp + cWeight * vertex_weight_smooth
    if np.sum(edge_weight_sharp) == 0:
        edge_weight_combined = cWeight * edge_weight_smooth
    else:
        edge_weight_combined = pWeight * edge_weight_sharp + cWeight * edge_weight_smooth

    return vertex_weight_combined, edge_weight_combined
