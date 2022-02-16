import numpy as np
import pyvista as pv
from PySubdiv.backend import optimization
from PySubdiv.backend import calculation
import math



def find_boundary_vertices(mesh, return_index=False, return_edge=False):
    """
    Find the coordinates of the vertices that are on the boundary in a 2D plane

    Parameters
    --------------
    mesh : PySubdiv mesh object
        mesh to find the boundary vertices
    return_index: bool
        if True the indices of the vertices are returned in addition
    return_edge: bool
        if True edges on the boundary are returned

    Returns
    --------------
    boundary_vertices: (n, 3) float
        Points in cartesian space
        n = number of boundary vertices
    index_boundary_vertices: n int
        Indices of the boundary vertices of the input mesh
    boundary_edges : (n, 2) int
        List of vertex indices making up the boundary_edges
    """

    if "edge_faces_dictionary" in mesh.data:
        pass
    else:
        mesh.edges_faces_connected()
    index_boundary_edges = []
    for edge_idx in mesh.data["edge_faces_dictionary"]:
        if len(mesh.data["edge_faces_dictionary"][edge_idx]) < 2:
            index_boundary_edges.append(edge_idx)
    boundary_edges = mesh.data['unique_edges'][index_boundary_edges]
    index_boundary_vertices = np.unique(boundary_edges)
    boundary_vertices = mesh.data['unique_edges'][index_boundary_vertices]

    if return_edge:
        if return_index:
            return index_boundary_vertices, boundary_vertices, boundary_edges
        else:
            return boundary_vertices, boundary_edges
    else:
        if return_index:
            return index_boundary_vertices, boundary_vertices, index_boundary_vertices,
        else:
            return boundary_vertices


def find_corner_vertices(mesh, index_boundary_vertices=None, return_index=False):
    """
    Find the coordinates of the vertices that form corners in a 2D plane by forming vectors between adjacent vertices
    calculating the vector angle. Angles between 45° and 135° are considered to be a corner of the plane.
    Returns corner vertices and boundary vertices. If corner vertices are needed it is not necessary to find boundary
    vertices beforehand

    Parameters
    --------------
    mesh : PySubdiv mesh object
        mesh to find the boundary vertices
    index_boundary_vertices: n int
        Indices of the boundary vertices of the input mesh. If no indices are passed they are found automatically
    return_index: bool
        if True the indices of the corner vertices are returned additionally

    Returns
    --------------
    corner_vertices: (n, 3) float
        Points in cartesian space of the corner vertices
        n = number of boundary vertices
    index_corner_vertices: n int
        Indices of the boundary vertices of the input mesh
    boundary_vertices: (n, 3) float
        Points in cartesian space of the boundary vertices
        n = number of boundary vertices
    index_boundary_vertices: n int
        Indices of the boundary vertices of the input mesh

    """

    if index_boundary_vertices is None:
        index_boundary_vertices = np.unique(find_boundary_vertices(mesh, return_index=True)[0])
    if "vertices_connected_dictionary" not in mesh.data:
        mesh.vertices_connected_dictionary()

    connected_vertices_matrix = mesh.data["vertices_connected_dictionary"]  # build vertex adjacency matrix
    num_boundary_vertices = len(index_boundary_vertices)  # number of vertices on the boundary of the mesh
    # create array to store connected vertices
    connected_vert_boundary_arr = np.ones((num_boundary_vertices, 4), dtype=int)
    connected_vert_boundary_arr[:, 0] = index_boundary_vertices
    connected_vert_boundary_arr[:, 3] = index_boundary_vertices
    corner_vert_mask = np.zeros(num_boundary_vertices, dtype=bool)  # index masks for corner and boundary vertices
    boundary_vert_mask = np.zeros(num_boundary_vertices, dtype=bool)

    for i in range(num_boundary_vertices):
        # index of the connected vertices to the i_th boundary vertex
        connected_vert = np.nonzero(connected_vertices_matrix[index_boundary_vertices[i]] == 1)[0]
        # check if the connected vertices are vertices on the boundary
        corner_vert_boundary = index_boundary_vertices[np.isin(index_boundary_vertices, connected_vert)]
        # if a vertex has only two adjacent vertices it can be used directly to compute vector angles
        if len(corner_vert_boundary) == 2:
            connected_vert_boundary_arr[i][1:3] = corner_vert_boundary
        # if it deviates from two the i_th vertex can still be a corner due to errors in the mesh. E.g zero area faces
        # on the boundary.
        else:
            boundary_vert_mask[i] = 1

    vector_1 = mesh.data['vertices'][connected_vert_boundary_arr[:, 0]] - mesh.data['vertices'][
        connected_vert_boundary_arr[:, 1]]
    vector_2 = mesh.data['vertices'][connected_vert_boundary_arr[:, 2]] - mesh.data['vertices'][
        connected_vert_boundary_arr[:, 3]]

    vector_1_unit_length = np.linalg.norm(vector_1, axis=1, keepdims=True)
    vector_2_unit_length = np.linalg.norm(vector_2, axis=1, keepdims=True)
    unit_vector_1 = vector_1 / vector_1_unit_length
    unit_vector_2 = vector_2 / vector_2_unit_length

    for i in range(len(unit_vector_1)):
        cos = abs(np.round(np.dot(unit_vector_1[i], unit_vector_2[i]), decimals=1))
        if cos <= 0.7:
            corner_vert_mask[i] = 1
        else:
            boundary_vert_mask[i] = 1

    index_corner_vertices = index_boundary_vertices[corner_vert_mask]
    index_boundary_vertices = index_boundary_vertices[boundary_vert_mask]
    corner_vertices = mesh.data['vertices'][index_corner_vertices]
    boundary_vertices = mesh.data['vertices'][index_boundary_vertices]

    if return_index:
        return corner_vertices, index_corner_vertices, boundary_vertices, index_boundary_vertices
    else:
        return corner_vertices, boundary_vertices


def find_bounds(mesh):
    """
    Find the bounds of the mesh or combination of meshes form is [x_min, x_max, y_min, y_max, z_min, z_max]

    Parameters
    --------------
    mesh: PySubdiv mesh

    Returns
    --------------
    bounds: list
        list of the bounds of the mesh in the form [x_min, x_max, y_min, y_max, z_min, z_max]

    """
    # check if list or single mesh is passed
    if isinstance(mesh, list):
        vertices = []
        for sub_mesh in mesh:
            vertices.append(sub_mesh.data['vertices'])
        vertices = np.concatenate(vertices)
    else:
        vertices = np.array(mesh.data['vertices'])

    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    z_max = np.max(vertices[:, 2])
    bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
    return bounds


def define_boundary_box(mesh, bounds=None):
    """
    Define the boundary box of a mesh or combination of meshes form
    Parameters
    --------------
    mesh: PySubdiv mesh

    bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
        bounds of a mesh. If None is passed bounds are found automatically

    Returns
    --------------
    boundary_box: (8,3) 8 vertices of the boundary box
    """

    if bounds is None:
        bounds = find_bounds(mesh)
    boundary_box = np.zeros((8, 3))
    counter = 0
    for x in bounds[:2]:
        for y in bounds[2:4]:
            for z in bounds[4:]:
                boundary_box[counter, 0] = x
                boundary_box[counter, 1] = y
                boundary_box[counter, 2] = z
                counter += 1
    return boundary_box


def find_mesh_for_fitting(vertices, mesh):
    """
    Three vertices which form a face are passed to the function. Tries to find the best mesh to fit the face to form the
    provided meshes in the list. Builds for each mesh a Kd-tree and finds and queries each vertex to find the closest
    point resp. the closest mesh to fit the vertices to. A raytrace is performed to find the closest mesh to the face
    formed by the three vertices. This is used to fit the new vertices which are created during subdivision
    --------------
    vertices: (n,3) float
        three vertices which forms a face
    mesh: list of PySubdiv meshes
        list of meshes to find the closest mesh for fitting

    Returns
    --------------
    mesh_id: int
        mesh index to which the face should be fitted to during subdivision
    """

    kd_trees = []
    index_nearest_mesh = []
    distance = []
    for part in mesh:
        kd_trees.append(optimization.build_tree(part))
    for vert in vertices:
        distance_nearest_point = []
        for tree in kd_trees:
            d = optimization.query_tree(vert, tree)[1]
            distance_nearest_point.append(d)
        index_nearest_mesh.append(np.where(distance_nearest_point == np.min(distance_nearest_point))[0])
    centroid = calculation.centroid(vertices)
    normal = calculation.face_normal(vertices)[0]
    for part in mesh:
        intersection_pos = calculation.intersection_point(centroid, normal, part)
        intersection_neg = calculation.intersection_point(centroid, -1 * normal, part)
        if len(intersection_pos) == 0:  # test if positive vector will find an ip
            if len(intersection_neg) == 0:  # test if negative vector will find an ip
                distance.append(math.inf)  # inf indicates now intersection resp. inf distance for both vectors
            else:
                distance.append(calculation.distance_vertices(centroid, intersection_neg))  # calculate dist of neg. vec
        else:  # positive vector results in ip
            if len(intersection_neg) == 0:  # test if negative vector will find an ip
                distance.append(calculation.distance_vertices(centroid, intersection_pos))  # calc dist of pos vec
            else:  # both vectors find an ip -> find minimal distance
                dist_pos = calculation.distance_vertices(centroid, intersection_pos)
                dist_neg = calculation.distance_vertices(centroid, intersection_neg)
                if dist_pos <= dist_neg:
                    distance.append(dist_pos)
                else:
                    distance.append(dist_neg)
    vert_mesh_fit = []  # index of mesh to fit the vertices to
    mesh_id = np.argmin(distance)
    for vert in index_nearest_mesh:
        if mesh_id in vert:
            vert_mesh_fit.append(mesh_id)
        else:
            vert_mesh_fit.append(vert[0])
    return mesh_id, vert_mesh_fit


def find_intersection_points(mesh_1, mesh_2, tol=None):
    """
    Finds intersection of two meshes by clipping the two meshes. The clipping will return the implicit distance field
    of the two meshes, with the parameter tol we can change the distance which is used to define the line of
    intersection

    --------------
    mesh_1 : PySubdiv mesh object
        first input mesh to check against second mesh.
    mesh_2 : PySubdiv mesh object
        second input mesh  to check against first mesh.
    tol : float
        tolerance for finding the line of intersection.
    Returns
    --------------
    intersection_points : (n,3) float
        x, y and z coordinates of the intersection point.
    """

    model_1 = mesh_1.model()  # create the two models to find line of intersection
    model_2 = mesh_2.model()
    intersection_1 = model_1.clip_surface(model_2, compute_distance=True)  # clip the surfaces with each other to find
    intersection_2 = model_2.clip_surface(model_1, compute_distance=True)  # the line of intersection

    # the tolerance is used to find the vertices which lies closely to the intersection line. If no tolerance is passed
    # the mean of the 2.5 % closest points is used to estimate the tolerance
    if tol is None:
        # concatenate the distances of the vertices to the intersection line.
        # Vertices with smaller distances are closer to the line of intersection
        concated_dist = abs(np.concatenate((intersection_1['implicit_distance'], intersection_2['implicit_distance'])))
        tol = np.mean(np.sort(concated_dist)[:int(0.025 * len(concated_dist))])

    intersection_points = [0]

    def change_tolerance(tolerance):

        intersection_points.pop()
        idx_1 = np.nonzero(abs(intersection_1['implicit_distance']) <= tolerance)[
            0]  # returning the indices of the vertices
        idx_2 = np.nonzero(abs(intersection_2['implicit_distance']) <= tolerance)[0]  # which lies in the tolerance

        point_cloud_intersection = pv.PolyData(intersection_1.points[idx_1]) + (
            pv.PolyData(intersection_2.points[idx_2]))
        if len(point_cloud_intersection.points) == 0:
            points = pl.add_mesh(pv.PolyData([0, 0, 0]), name='point_cloud_intersection', render_points_as_spheres=True)
            pl.remove_actor(points)
        else:
            points = pl.add_mesh(point_cloud_intersection, name='point_cloud_intersection',
                                 render_points_as_spheres=True)
        intersection_points.append(np.concatenate((intersection_1.points[idx_1], intersection_2.points[idx_2])))
        print('test', intersection_points)
        return

    pl = pv.Plotter()
    pl.add_slider_widget(change_tolerance, [0, tol + 1], value=tol, title='tolerance')
    pl.add_text('Use slider to change tolerance', position='lower_edge', name='textbox')
    pl.set_background('royalblue', top='aliceblue')
    pl.add_mesh(model_1, style='wireframe', color='green')
    pl.add_mesh(model_2, style='wireframe', color='red')
    pl.show()
    return intersection_points[0]


