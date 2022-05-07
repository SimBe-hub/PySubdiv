import networkx as nx
import numpy as np
from PySubdiv.backend import calculation
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import math


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


def vertex_vertex_dictionary(vertices, edges):
    """
    Returns adjacency of vertices in a dictionary.

    Parameters
    ----------
    vertices: (n, 3) float
        Vertices of the mesh.
    edges: (n, 2) int
        Edges of the mesh.

    Returns
    ---------
    vertex_vertex_dict : dict
        showing which vertex is connected to which vertex. Key vertex index, values connected vertices
    """

    vertex_vertex_dict = {index: [] for index in range(len(vertices))}
    for index_edge, vertex in enumerate(edges[:, 0]):
        if edges[index_edge][1] not in vertex_vertex_dict[vertex]:
            vertex_vertex_dict[vertex].append(edges[index_edge][1])
    for index_edge, vertex in enumerate(edges[:, 1]):
        if edges[index_edge][1] not in vertex_vertex_dict[vertex]:
            vertex_vertex_dict[vertex].append(edges[index_edge][0])
    return vertex_vertex_dict


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


def vertex_edges_dict(vertices, edges):
    """
    Returns incidence of vertex edges. The key in the dictionary is assigned to a vertex and the values are the indices
    of the connected edges.

    Parameters
    ----------
    vertices: (n ,3) float
        List of vertices from the mesh.
    edges: (n, 2) int
        List of vertex indices connected to form the edges

    Returns
    ---------
    vertex_edges_dict : dict
        Dictionary of vertices (keys) and connected edges (values)
    """
    vertex_edges_dict = {index: [] for index in range(len(vertices))}
    for edgeIndex, edge in enumerate(edges):
        vertex_edges_dict[edge[0]].append(edgeIndex)
        vertex_edges_dict[edge[1]].append(edgeIndex)
    for key in vertex_edges_dict:
        vertex_edges_dict[key] = np.array(vertex_edges_dict[key])
    return vertex_edges_dict


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


def edge_faces_dict(unique_edges, sorted_edges):
    """
    Returns incidence edges and faces. The dictionary key is the edge index and the values are the face indices.

    Parameters
    ----------
    unique_edges: (n, 2) int
        List of vertex indices connected to form the edges with removed duplicates
    sorted_edges: (n, 2) int
        List of vertex indices connected to form the edges, sorted along axis 1

    Returns
    ---------
    edge_face_dict : dict
        incidence of faces and edges in dictionary. Key is the edge index and the values are the face indices.
    """
    unique_edges_dict = {}
    for counter, unique_edge in enumerate(unique_edges):
        unique_edges_dict[tuple(unique_edge)] = counter
    edge_index = []
    for edge in sorted_edges:
        if tuple(edge) in unique_edges_dict:
            edge_index.append(unique_edges_dict[tuple(edge)])
        else:
            edge_index.append(unique_edges_dict[tuple(edge[::-1])])

    # every three edges belongs to a face. Increment the face index every third ege.
    face_index = 0
    face_index_counter = 0
    edge_face_dict = {index: [] for index in edge_index}

    for index in edge_index:
        edge_face_dict[index].append(face_index)
        face_index_counter += 1
        if face_index_counter == 3:
            face_index += 1
            face_index_counter = 0

    return edge_face_dict


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


def vertex_faces_dict(faces, vertices):
    """
    Returns incidence of vertices and faces in a dictionary. Each key is assigned to a vertex and value is the connected
    face index

    Parameters
    ----------
    faces: (n, 3) int
        Triangular faces of the mesh
    vertices: (n, 3) float
        Vertices of the mesh


    Returns
    ---------
    vertex_face_dictionary : dict
        incidence of vertices and faces
    """

    vertex_face_dict = {index: [] for index in range(len(vertices))}
    for faceIndex, face in enumerate(faces):
        vertex_face_dict[face[0]].append(faceIndex)
        vertex_face_dict[face[1]].append(faceIndex)
        vertex_face_dict[face[2]].append(faceIndex)
    for key in vertex_face_dict:
        vertex_face_dict[key] = np.array(vertex_face_dict[key])

    return vertex_face_dict


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
