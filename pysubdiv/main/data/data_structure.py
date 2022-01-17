import networkx as nx
import numpy as np
from pysubdiv.backend import calculation
from scipy.sparse import coo_matrix
import math
import time

def mesh_graph(vertices, edges):
    """
    Returns a networkx graph representing the vertices and
    their connections in the mesh.

    Parameters
    ----------
    vertices: (n, 3) float
        Vertices of the mesh defining the nodes of the graph structure
    edges: (n, 2) int
        List of vertex indices connected to form the edges

    Returns
    ---------
    graph : networkx.Graph
        Graph structure of vertices and edges
    """
    g = nx.Graph()
    nodes = np.arange(len(vertices))
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    return g


def adjacent_vertices(graph, vertices, matrix_form=False):
    """
    Returns adjacency matrix of the graph. Show connected vertices in list or matrix form.

    Parameters
    ----------
    graph: networkx.Graph
    vertices: (n, 3) float
        Vertices of the mesh defining the nodes of the graph structure
    matrix_form: bool
        When True adjacency matrix is passed as matrix when False as list

    Returns
    ---------
    graph : networkx.Graph.adjacency_matrix
        Adjacency matrix of connected vertices
    """

    if matrix_form:
        return nx.adjacency_matrix(graph, nodelist=np.arange(len(vertices))).toarray()
    else:
        return nx.adjacency_matrix(graph, nodelist=np.arange(len(vertices)))


def vert_edge_incident(graph, edges, vertices):
    """
    Returns incidence matrix of the vertex-edge graph. Each row is assigned to a vertex and each column to an edge

    Parameters
    ----------
    graph: networkx.Graph
    vertices: (n, 3) float
        Vertices of the mesh defining the nodes of the graph structure
    edges: (n, 2) int
        List of vertex indices connected to form the edges

    Returns
    ---------
    graph : Graph.incidence_matrix
        incidence Matrix of vertices and edges
    """

    return nx.incidence_matrix(graph, edgelist=edges,
                               nodelist=np.arange(len(vertices)))


def face_edges_incident(unique_edges, sorted_edges, faces, mesh_type):
    """
    Returns incidence matrix of faces and edges. Each row is assigned to a face and each column to an edge.

    Parameters
    ----------
    unique_edges: (n, 2) int
        List of vertex indices connected to form the edges with removed duplicates
    sorted_edges: (n, 2) int
        List of vertex indices connected to form the edges, sorted along axis 1

    faces: (n, 3) or (n, 4) int
        Triangular or quadrilateral faces of the mesh
    mesh_type : str
        type of the mesh: triangular or quadrilateral

    Returns
    ---------
    graph : scipy.coo_matrix
        incidence matrix of faces and edges
    """
    unique_edges_dict = {}
    counter = 0
    for unique_edge in unique_edges:
        unique_edges_dict[tuple(unique_edge)] = counter
        counter += 1
    col = []

    for edge in sorted_edges:
        if tuple(edge) in unique_edges_dict:
            col.append(unique_edges_dict[tuple(edge)])
        else:
            col.append(unique_edges_dict[tuple(edge[::-1])])

    row = calculation.faces_to_edges(faces, mesh_type, return_index=True)[1]
    data = np.ones(len(col))
    coo = coo_matrix((data, (row, col)), shape=(len(faces), len(unique_edges))).toarray()
    return coo


def vertex_face_incident(faces, vertices):
    """
    Returns incidence matrix of vertices and faces. Each row is assigned to a vertex and each column to a face

    Parameters
    ----------
    faces: (n, 4) int
        Quadrilateral faces of the mesh
    vertices: (n, 3) float
        Vertices of the mesh


    Returns
    ---------
    graph : scipy.coo_matrix
        incidence matrix of vertices and faces
    """
    row = []
    col = []
    arr_faces = np.array(faces)

    for i in range(len(vertices)):
        vertex_edge_connected = np.nonzero(arr_faces == i)[0]  # vertex connected to which face (face indices)
        col.extend(vertex_edge_connected)  # flattend list of connections
        n = len(vertex_edge_connected)  # how many connected faces to a vertex
        row.extend(n * [i])
    data = np.ones(len(col))
    coo = coo_matrix((data, (row, col)), shape=(len(vertices), len(faces))).toarray()
    return coo


def get_edge_face_connectivity(faces_edges_incidence, edge_index):
    connected_faces = np.nonzero(faces_edges_incidence[:, edge_index] == 1)[0]
    return connected_faces


def laplace_matrix(vertices, faces, vertex_face_incidence, faces_edge_incidence, edges, vertex_edge_incidence):
    degree_matrix = np.zeros((len(vertices), len(vertices)))
    weight_matrix = np.zeros((len(vertices), len(vertices)))

    # array of the face indices repeated for every edge
    face_indices = np.tile(np.arange(len(faces)), len(edges))

    # boolean array where connected tris are True; taken from faces_edge_matrix
    connected_tris = (faces_edge_incidence[:, range(len(edges))] == 1).T.flatten()

    # masking the arrays to get only the connected triangular indices
    connected_tris_masked = face_indices[connected_tris].reshape(-1, 2)
    # finding vertices opposite to the edge
    opposite_verts = np.zeros((len(connected_tris_masked), 2), dtype=int)

    for i in range(len(connected_tris_masked)):
        opposite_verts[i] = np.nonzero(np.sum(vertex_face_incidence[:, connected_tris_masked[i]], axis=1) == 1)[0]

    for i in range(len(degree_matrix)):
        degree_matrix[i][i] = np.sum(vertex_face_incidence[i])
        vertex_edge = np.nonzero(vertex_edge_incidence[i] == 1)[0]
        counter = 0
        for connected_edge in vertex_edge:
            edge = edges[connected_edge]
            footpoint = opposite_verts[vertex_edge[counter]]
            if edge[0] == i:
                angle_1 = calculation.angle_vertices(vertices[i], vertices[edge[1]], vertices[footpoint[0]])
                angle_2 = calculation.angle_vertices(vertices[i], vertices[edge[1]], vertices[footpoint[1]])
                connected_vert = edge[1]
            else:
                angle_1 = calculation.angle_vertices(vertices[i], vertices[edge[0]], vertices[footpoint[0]])
                angle_2 = calculation.angle_vertices(vertices[i], vertices[edge[0]], vertices[footpoint[1]])
                connected_vert = edge[0]

            weight_matrix[i][connected_vert] = 0.5 * ((1 / math.tan(angle_1)) + (1 / math.tan(angle_2)))
            counter += 1

    return weight_matrix - degree_matrix


def matrix_to_list(matrix, invert=False):
    incidence_list = []
    if not invert:
        for i in range(len(matrix[:, 0])):
            incidence_list.append(np.nonzero(matrix[i, :] == 1)[0].tolist())
    else:
        for i in range(len(matrix[0, :])):
            incidence_list.append(np.nonzero(matrix[:, i] == 1)[0].tolist())

    return incidence_list
