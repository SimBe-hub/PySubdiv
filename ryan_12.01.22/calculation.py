import numpy as np
import Catmull_Clark
import loop
import loop_optimization
import simple_subidivsion_triangles
import copy
import utils


def faces_to_edges(faces, mesh_type, return_index=False):
    """
    Given a numpy.ndarray of faces with the shape (n,4) or (n,3), return a numpy.ndarray of edges (n*4,2) resp (n*3, 2)

    Parameters
    -----------
    faces : np.ndarray (n, 4) or (n,3) int
        Vertex indices representing faces
    mesh_type: string
        quadrilateral or triangular mesh

    return_index: bool
        When True returns indices of the faces for the corresponding edge

    Returns
    -----------
    edges : np.ndarray (n*4, 2) resp. (n*3 , 2) int
      Vertex indices representing edges
    """

    # Pairs of vertex indices building the edges are extracted from the faces
    if mesh_type == 'quadrilateral':
        edges_per_face = 4
        edges = faces[:, [0, 1, 1, 2, 2, 3, 3, 0]].reshape((-1, 2))
    else:
        edges_per_face = 3
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    if return_index:
        # edges are in order of faces due to reshape
        face_index = np.tile(np.arange(len(faces)),
                             (edges_per_face, 1)).T.reshape(-1)
        return edges, face_index
    return edges


def call_subdiv_algorithm(mesh, level, interactive=False, simple_subdivision=False, for_optimization=False):
    """
    Calls the Catmull_Clark module, loop module or simple subdivision module and performs the subdivision algorithm
    on the provided mesh.
    Will return the calculated vertices and faces.

    Parameters
    ----------
    mesh : PySubdiv mesh
    level: int
        How many times the subdivision is performed.
    interactive: bool
        Sets the provided mesh either as interactive or non-interactive
    for_optimization: bool
        Sets the provided mesh for optimization purposes, only one level of subdivision. No vertex connectivity
         calculated
    Returns
    ---------
    Mesh:
        New mesh with the subdivided vertices and faces.

    """

    if interactive:
        mesh.data['interactive'] = True
        if mesh.data['mesh_type'] == 'quadrilateral':
            surf = Catmull_Clark.Structure(mesh)
        else:
            surf = loop.Structure(mesh)
        for i in range(level):
            surf.main()
        return surf
    else:
        mesh.data['interactive'] = False
        if mesh.data['mesh_type'] == 'quadrilateral':
            surf = Catmull_Clark.Structure(mesh)
        else:
            if not simple_subdivision:
                if for_optimization:
                    surf = loop_optimization.Structure(mesh)
                else:
                    surf = loop.Structure(mesh)
            else:
                surf = simple_subidivsion_triangles.Structure(mesh)

        for i in range(level):
            surf.main()
    return surf.data


def intersection_point(footpoint, vector, mesh, multiplicator=100, first_point=True):
    """
    Cast a ray from a footpoint along a vector and find the intersection point between a mesh

    Parameters
    ----------
    footpoint : [x, y , z] float
        coordinate of the footpoint
    vector : [x ,y, z] float
        vector along the ray is casted
    mesh : PySubdiv mesh
        mesh on which to find the intersection
    Returns
    ---------
    ip : [x, y, z] float
        Coordinates of the intersection between the ray and the mesh
    """
    poly_mesh = mesh.model()  # convert mesh to PyVista.PolyData to perform ray cast on
    vec = footpoint + vector * multiplicator  # set length of the ray
    ip = poly_mesh.ray_trace(footpoint, vec, first_point=first_point, plot=False)[0]
    return ip


def distance_vertices(vertex_1, vertex_2):
    """
    Calculate distance between two vertices.

    Parameters
    ----------
    vertex_1 : [x, y , z] float
        coordinate of the first vertex
    vertex_2 : [x ,y, z] float
        coordinate of the second vertex
    Returns
    ---------
    distance :  float
        distance between the two vertices
    """
    vector = vertex_1 - vertex_2
    distance = np.linalg.norm(vector)
    return distance


def center_of_mass(mesh):
    """
    Calculate the center of mass of a mesh assuming uniform mass distribution

    Parameters
    ----------
    mesh : PySubdiv mesh

    Returns
    ---------
    com :  float
        center of mass of the mesh
    """
    com = np.mean(mesh.vertices, axis=0)
    return com


def intersection_point_old(mesh, idx_1, idx_2, plot=False):
    """
    Find and calculate the intersection point between a mesh and two vertices above and under the mesh

    Parameters
    ----------
    idx_1 : index of the first vertex
    idx_2 : index of the second vertex
    mesh : PySubdiv mesh

    Returns
    ---------
    ip : [x, y, z]
        Coordinates of the intersection between two vertices and the mesh.

    """

    poly_mesh = mesh.model()
    p0 = copy.copy(mesh.data['vertices'][idx_1])
    p1 = copy.copy(mesh.data['vertices'][idx_2])
    ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True, plot=plot)
    displacement_list = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0, rtol=1e-03,
                                                                              atol=1e-05):
        connected_verts_p0 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_1] == 1)[0]
        connected_verts_p1 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_2] == 1)[0]
        shared_connection = mesh.data['vertices'][list(set(connected_verts_p0).intersection(connected_verts_p1))]
        for displacement in displacement_list:
            # connected_verts_p0 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_1] == 1)[0]
            # connected_verts_p1 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_2] == 1)[0]
            # shared_connection = mesh.data['vertices'][list(set(connected_verts_p0).intersection(connected_verts_p1))]
            vector_p0 = - (p0 - shared_connection)
            vector_p1 = - (p1 - shared_connection)
            p0 += vector_p0[0] * displacement
            p1 += vector_p1[0] * displacement
            ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True, plot=plot)
            if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0, rtol=1e-03,
                                                                                      atol=1e-05):
                p0 = copy.copy(mesh.data['vertices'][idx_1])
                p1 = copy.copy(mesh.data['vertices'][idx_2])
            else:
                intersection = ip
                return intersection

    else:
        intersection = ip
        return intersection


def intersection_idx_point(mesh, idx, point_2, boundary_vertices_indices, connected_vert, plot=False):
    """
    Find and calculate the intersection point between a mesh and two vertices above and under the mesh. The second point
    is passed as coordinates

    Parameters
    ----------
    idx : index of the first vertex
    point_2 : coordinates of second vertices
    mesh : PySubdiv mesh

    Returns
    ---------
    ip : [x, y, z]
        Coordinates of the intersection between two vertices and the mesh.

    """

    poly_mesh = mesh.model()
    p0 = copy.copy(mesh.data['vertices'][idx])
    p1 = copy.copy(point_2)
    ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True)
    displacement_list = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # displacement_list = [1, 2]
    if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0, rtol=1e-03,
                                                                              atol=1e-05):
        connected_verts_p0 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx] == 1)[0].tolist()
        del connected_verts_p0[np.nonzero(connected_verts_p0 == connected_vert)[0][0]]
        for idx_2 in connected_verts_p0:
            if idx_2 in boundary_vertices_indices:
                for displacement in displacement_list:
                    vector = -(p0 - mesh.data['vertices'][idx_2])
                    p0 += vector * displacement
                    p1 += vector * displacement
                    ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True)

                    if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0,
                                                                                              rtol=1e-03,
                                                                                              atol=1e-05):
                        p0 = copy.copy(mesh.data['vertices'][idx])
                        p1 = copy.copy(point_2)
                    else:
                        intersection = ip
                        return intersection
    else:
        return ip


def centroid(vertices):
    """
    calculate the centroid_tris of a triangular face.

    Parameters
    ----------
    vertices : (3,3) float
        vertices forming a triangular face on which the centroid_tris should be calculated

    Returns
    ---------
    centroid_tris : [x, y, z]

    """
    centroid_tris = np.sum(vertices, axis=0) / 3
    return centroid_tris


def face_normal(vertices):
    """
    Calculate the normal of a face formed by three vertices

    Parameters
    ----------
    vertices : (n,3,3) float
        vertices forming a triangular face on which the face normal should be calculated
    Returns
    ---------
    face_normal_vector : [x, y, z] array
        normal vector of the face
    """
    vertices = np.array(vertices)
    if utils.is_shape(vertices, (-1, 3, 3)):
        pass
    else:
        vertices = vertices.reshape((-1, 3, 3))

    vector1 = vertices[:, 1] - vertices[:, 0]
    vector2 = vertices[:, 2] - vertices[:, 0]
    normal_vector = np.cross(vector1, vector2)
    normal_vector_unit_length = normal_vector / np.linalg.norm(normal_vector, axis=1, keepdims=True)
    return normal_vector_unit_length


def ray_cast(point, direction, target_mesh, back_cast=False, plot=False):

    long_ray = copy.copy(direction)
    long_ray *= 100
    long_ray += point
    if not back_cast:
        intersection_points_long, cells = target_mesh.ray_trace(point, long_ray, first_point=False, plot=plot)
    else:
        intersection_points_long, cells = target_mesh.ray_trace(long_ray, point, first_point=False, plot=plot)
    return intersection_points_long, cells


def angle_vertices(vertex_1, vertex_2, footpoint_vertex):
    """
    Calculate the angle between two vertices and footpoint

    Parameters
    ----------
    vertex_1 : (3,3) float
        coordinate of the first vertex
    vertex_2 : (3,3) float
        coordinate of the second vertex
    footpoint_vertex : (3,3) float
        coordinate of the footpoint vertex
    Returns
    ---------
    angle : float
        angle in radians
    """
    vector1 = vertex_1 - footpoint_vertex
    vector2 = vertex_2 - footpoint_vertex
    vector1_unit = vector1 / np.linalg.norm(vector1)
    vector2_unit = vector2 / np.linalg.norm(vector2)
    # angle = np.arccos(np.dot(vector1_unit, vector2_unit)) * 180 / np.pi
    angle = np.arccos(np.dot(vector1_unit, vector2_unit))  # radians
    return angle


def mesh_dimension(mesh):
    xyz_max = []
    xyz_min = []
    for i in range(3):
        xyz_max.append(np.max(mesh.vertices[:, i]).tolist())
        xyz_min.append(np.min(mesh.vertices[:, i]).tolist())
    return xyz_max, xyz_min


def angle_around_vertex(mesh, vertex_idx):
    connected_verts = np.nonzero(mesh.data['vertices_adjacency_matrix'][vertex_idx] == 1)[0]
    number_connected_verts = len(connected_verts)
    position_vector_vertex = np.tile(mesh.data['vertices'][vertex_idx], (number_connected_verts, 1))
    position_vector_connected_verts = mesh.data['vertices'][connected_verts]
    vector = position_vector_vertex - position_vector_connected_verts
    vector_unit_lenght = np.linalg.norm(vector, axis=1, keepdims=True)
    unit_vector = vector / vector_unit_lenght
    angle = 0
    counter = []
    for i in range(len(unit_vector)):
        for j in range(len(unit_vector)):
            if i == j:
                continue
            elif hash(frozenset([i, j])) in counter:
                continue
            else:
                counter.append(hash(frozenset([i, j])))
                angle += np.arccos(np.dot(unit_vector[i], unit_vector[j])) * 180 / np.pi
    return angle
