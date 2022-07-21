# indices_static_vertices = np.nonzero(self.control_mesh.data['dynamic_vertices'] == 's')[0]
            # test_list = []
            # self.control_mesh.recalculate_face_normals()
            # #print(indices_static_vertices)
            # for index_static_vertex in indices_static_vertices[:2]:
            #     adjacent_vertices_index = np.nonzero(
            #         self.control_mesh.data['vertices_adjacency_matrix'][index_static_vertex] == 1)
            #     #print(adjacent_vertices_index)
            #     adjacent_vertices = self.control_mesh.vertices[adjacent_vertices_index]
            #     #print(adjacent_vertices)
            #     adjacent_vertices_mean = np.mean(adjacent_vertices, axis=0)
            #     test_list.append(adjacent_vertices_mean)
            #     angle_list = []
            #     for adjacent_vertex in adjacent_vertices:
            #         angle_list.append(abs((np.dot(self.control_mesh.vertex_normals[index_static_vertex],
            #                                      adjacent_vertex))))
            #     print(angle_list)
            #     print(self.control_mesh.vertices[index_static_vertex])
            #     print(adjacent_vertices_mean)
            #     #self.control_mesh.vertices[index_static_vertex] = adjacent_vertices_mean
            # for count, vertex_index in enumerate(indices_static_vertices[:2]):
            #     self.control_mesh.vertices[vertex_index] = test_list[count]
            #self.initialize_v()
            #self._control_cage.vertices[self.variable_vertices_idx] = self.p
            #self._control_cage.set_crease(minimized_creases, self.variable_edges)
            #self.z = result_z


# def intersection_point_old(mesh, idx_1, idx_2, plot=False):
#     """
#     Find and calculate the intersection point between a mesh and two vertices above and under the mesh
#
#     Parameters
#     ----------
#     idx_1 : index of the first vertex
#     idx_2 : index of the second vertex
#     mesh : PySubdiv mesh
#
#     Returns
#     ---------
#     ip : [x, y, z]
#         Coordinates of the intersection between two vertices and the mesh.
#
#     """
#
#     poly_mesh = mesh.model()
#     p0 = copy.copy(mesh.data['vertices'][idx_1])
#     p1 = copy.copy(mesh.data['vertices'][idx_2])
#     ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True, plot=plot)
#     displacement_list = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
#
#     if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0, rtol=1e-03,
#                                                                               atol=1e-05):
#         connected_verts_p0 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_1] == 1)[0]
#         connected_verts_p1 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_2] == 1)[0]
#         shared_connection = mesh.data['vertices'][list(set(connected_verts_p0).intersection(connected_verts_p1))]
#         for displacement in displacement_list:
#             # connected_verts_p0 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_1] == 1)[0]
#             # connected_verts_p1 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx_2] == 1)[0]
#             # shared_connection = mesh.data['vertices'][list(set(connected_verts_p0).intersection(connected_verts_p1))]
#             vector_p0 = - (p0 - shared_connection)
#             vector_p1 = - (p1 - shared_connection)
#             p0 += vector_p0[0] * displacement
#             p1 += vector_p1[0] * displacement
#             ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True, plot=plot)
#             if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0, rtol=1e-03,
#                                                                                       atol=1e-05):
#                 p0 = copy.copy(mesh.data['vertices'][idx_1])
#                 p1 = copy.copy(mesh.data['vertices'][idx_2])
#             else:
#                 intersection = ip
#                 return intersection
#
#     else:
#         intersection = ip
#         return intersection


# def intersection_idx_point(mesh, idx, point_2, boundary_vertices_indices, connected_vert, plot=False):
#     """
#     Find and calculate the intersection point between a mesh and two vertices above and under the mesh. The second point
#     is passed as coordinates
#
#     Parameters
#     ----------
#     idx : index of the first vertex
#     point_2 : coordinates of second vertices
#     mesh : PySubdiv mesh
#
#     Returns
#     ---------
#     ip : [x, y, z]
#         Coordinates of the intersection between two vertices and the mesh.
#
#     """
#
#     poly_mesh = mesh.model()
#     p0 = copy.copy(mesh.data['vertices'][idx])
#     p1 = copy.copy(point_2)
#     ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True)
#     displacement_list = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#     # displacement_list = [1, 2]
#     if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0, rtol=1e-03,
#                                                                               atol=1e-05):
#         connected_verts_p0 = np.nonzero(mesh.data['vertices_adjacency_matrix'][idx] == 1)[0].tolist()
#         del connected_verts_p0[np.nonzero(connected_verts_p0 == connected_vert)[0][0]]
#         for idx_2 in connected_verts_p0:
#             if idx_2 in boundary_vertices_indices:
#                 for displacement in displacement_list:
#                     vector = -(p0 - mesh.data['vertices'][idx_2])
#                     p0 += vector * displacement
#                     p1 += vector * displacement
#                     ip, ic = poly_mesh.ray_trace(p0, p1, first_point=True)
#
#                     if np.allclose(np.sum(p0 - ip), 0, rtol=1e-03, atol=1e-05) or np.allclose(np.sum(p1 - ip), 0,
#                                                                                               rtol=1e-03,
#                                                                                               atol=1e-05):
#                         p0 = copy.copy(mesh.data['vertices'][idx])
#                         p1 = copy.copy(point_2)
#                     else:
#                         intersection = ip
#                         return intersection
#     else:
#         return ip


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
