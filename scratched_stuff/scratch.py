# indices_static_vertices = np.nonzero(self.control_cage.data['dynamic_vertices'] == 's')[0]
            # test_list = []
            # self.control_cage.recalculate_face_normals()
            # #print(indices_static_vertices)
            # for index_static_vertex in indices_static_vertices[:2]:
            #     adjacent_vertices_index = np.nonzero(
            #         self.control_cage.data['vertices_adjacency_matrix'][index_static_vertex] == 1)
            #     #print(adjacent_vertices_index)
            #     adjacent_vertices = self.control_cage.vertices[adjacent_vertices_index]
            #     #print(adjacent_vertices)
            #     adjacent_vertices_mean = np.mean(adjacent_vertices, axis=0)
            #     test_list.append(adjacent_vertices_mean)
            #     angle_list = []
            #     for adjacent_vertex in adjacent_vertices:
            #         angle_list.append(abs((np.dot(self.control_cage.vertex_normals[index_static_vertex],
            #                                      adjacent_vertex))))
            #     print(angle_list)
            #     print(self.control_cage.vertices[index_static_vertex])
            #     print(adjacent_vertices_mean)
            #     #self.control_cage.vertices[index_static_vertex] = adjacent_vertices_mean
            # for count, vertex_index in enumerate(indices_static_vertices[:2]):
            #     self.control_cage.vertices[vertex_index] = test_list[count]
            #self.initialize_v()
            #self._control_cage.vertices[self.variable_vertices_idx] = self.p
            #self._control_cage.set_crease(minimized_creases, self.variable_edges)
            #self.z = result_z