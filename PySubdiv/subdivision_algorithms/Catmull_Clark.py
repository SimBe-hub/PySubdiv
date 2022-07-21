import numpy as np
from PySubdiv.backend import calculation
from PySubdiv.data import data_structure, data


class Structure(object):

    def __init__(self, mesh):

        self.data = data.data_dictionary()
        self.data['vertices'] = mesh.data["vertices"].tolist()
        self.data['faces'] = mesh.data["faces"]
        self.edges_list = []
        self.creases_list = []
        self.data['mesh_type'] = mesh.data['mesh_type']

        if 'unique_edges' in mesh.data:
            self.data['unique_edges'] = mesh.data["unique_edges"]
        else:
            self.data['unique_edges'] = mesh.edges_unique()

        if 'creases' in mesh.data:
            self.data['creases'] = mesh.data['creases']
        else:
            self.data['creases'] = mesh.set_crease()

        if 'faces_edges_matrix' in mesh.data:
            self.data['faces_edges_matrix'] = mesh.data['faces_edges_matrix']
        else:
            self.data['faces_edges_matrix'] = mesh.faces_edges_incidence()
        if 'vert_edge_list' in mesh.data:
            self.data['vert_edge_list'] = mesh.data['vert_edge_list']
        else:
            self.data['vert_edge_list'] = mesh.vertices_edges_incidence()
        if 'vertices_adjacency_matrix' in mesh.data:
            self.data['vertices_adjacency_matrix'] = mesh.data['vertices_adjacency_matrix']
        else:
            self.data['vertices_adjacency_matrix'] = mesh.vertices_connected()
        if 'vertex_faces_matrix' in mesh.data:
            self.data['vertex_faces_matrix'] = mesh.data['vertex_faces_matrix']
        else:
            self.data['vertex_faces_matrix'] = mesh.vertices_faces_incidence()

        if 'ctr_points' in mesh.data:
            self.data['ctr_points'] = mesh.data['ctr_points']

        else:
            self.data['ctr_points'] = mesh.ctr_points(self.data['vertices'])

        if mesh.data['interactive']:
            self._interactive = True
            self.data['vertices'][:len(self.data['ctr_points'])] = self.data['ctr_points']

        else:
            self._interactive = False

        self._vertices_edges = []

        self.edges_list.append(self.data['unique_edges'])
        self.creases_list.append(self.data['creases'])

    def add_vertex(self, v):
        self.data['vertices'].append(v)
        return len(self.data['vertices']) - 1

    def add_vertex_multiple(self, v):
        start = len(self.data['vertices'])
        self.data['vertices'].extend(v)
        return np.arange(start, len(self.data['vertices']))

    def quads_contain(self, *indices):
        connected = []
        # print(indices)

        # For every quad
        for curQuad, quadVerts in enumerate(self.data['faces'], 0):

            # If every indice exists in the quad, add it to our list
            for idx in indices:
                if idx not in quadVerts:
                    break
            else:
                connected.append(curQuad)

        return connected

    def midpoint(self, *indices):
        mpoint = np.array([0, 0, 0])
        for idx in indices:
            mpoint = mpoint + self.data['vertices'][idx]
        mpoint = mpoint / (len(indices))
        return mpoint

    def midpoint_multiple(self, idx_arr0, idx_arr1, idx_arr2=None, idx_arr3=None):

        vertices = np.asanyarray(self.data['vertices'])
        if idx_arr2 is None and idx_arr3 is None:
            mpoint = (np.add(vertices[[idx_arr0]], vertices[[idx_arr1]])) / 2

        else:
            add_1 = np.add(vertices[[idx_arr0]], vertices[[idx_arr1]])
            add_2 = np.add(vertices[[idx_arr2]], vertices[[idx_arr3]])
            mpoint = np.add(add_1, add_2) / 4
        return mpoint

    def midpoint_edges_old(self, *indices):
        mpoint = np.array([0, 0, 0])
        for idx in indices:
            mpoint = mpoint + self.data['vertices'][idx]
        mpoint = mpoint / (len(indices))
        return mpoint

    def midpoint_edges(self, indices, weights):

        mpoint = np.array([0, 0, 0])
        for i in range(len(indices)):
            mpoint = mpoint + self.data['vertices'][indices[i]] * weights[i]
        return mpoint

    def fractional_weight(self, index_incident_edges):
        transition_count = 0
        transition_sum = 0
        for idx in index_incident_edges:
            transition_sum += self.data['creases'][idx]
            if self.data['creases'][idx] > 0:
                transition_count += 1

        fractional_weight = transition_sum / transition_count
        return fractional_weight

    def generate_face_points(self):
        faces_arr = np.asanyarray(self.data['faces'])  # convert list of faces to np.array
        verts_arr = np.asanyarray(self.data['vertices'])  # convert list of vertices to np.array
        # calculate coordinates of face points
        face_points = (np.sum(verts_arr[faces_arr], axis=1) / len(faces_arr[0])).tolist()

        quad_ctr_list = self.add_vertex_multiple(face_points)  # List of indices of the facepoints in the vertices array
        return quad_ctr_list

    def generate_edge_points(self, quad_ctr_list):
        # np.array to store vertices of the edge points
        edge_verts = np.zeros((len(self.data['unique_edges']), 3))

        # seperate boundary odd vertices from edge vertices

        idx_boundary_odd_edges = np.nonzero(np.sum(self.data['faces_edges_matrix'], axis=0) <= 1)[0]
        idx_non_manifold_edges = np.nonzero(np.sum(self.data['faces_edges_matrix'], axis=0) >= 3)[0]
        idx_non_manifold_verts = np.unique(self.data['unique_edges'][idx_non_manifold_edges])

        idx_edge_verts = np.nonzero((np.sum(self.data['faces_edges_matrix'], axis=0) > 1) &
                                    (np.sum(self.data['faces_edges_matrix'], axis=0) < 3))[0]
        list_idx = [idx_boundary_odd_edges, idx_edge_verts]
        boundary_odd_verts = True
        for indices in list_idx:
            idx_zero_crease = indices[np.nonzero(self.data['creases'][indices] == 0)[0]]
            # array with indices of unique_edges where crease == 1
            idx_infinite_crease = indices[np.nonzero(self.data['creases'][indices] == 1)[0]]
            # array with indices of unique_edges where crease > 0 & < 1
            idx_crease = indices[np.nonzero((self.data['creases'][indices] > 0)
                                            & (self.data['creases'][indices] < 1))[0]]

            if boundary_odd_verts:

                boundary_odd_verts = False
                # calculating the position for boundary odd vertices with zero crease #part must be revisited
                edge_verts[idx_boundary_odd_edges] = self.midpoint_multiple(  # creases for boundary odd vertices might
                    self.data['unique_edges'][idx_boundary_odd_edges][:, 0],  # not work correctly
                    self.data['unique_edges'][idx_boundary_odd_edges][:, 1])

                edge_verts[idx_non_manifold_edges] = self.midpoint_multiple(  # can be put together with boundary odd
                    self.data['unique_edges'][idx_non_manifold_edges][:, 0],  # vertices to shorten the code
                    self.data['unique_edges'][idx_non_manifold_edges][:, 1])

            else:
                # array of the face indices repeated for every index in idx_zero_crease
                arr_face_indices_zero_crease = np.tile(np.arange(len(self.data['faces'])), len(idx_zero_crease))
                # boolean array where connected quads with zero crease are True; taken from faces_edge_matrix
                connected_quads_zero_crease = (self.data['faces_edges_matrix'][:, idx_zero_crease] == 1).T.flatten()
                # masking the arrays to get only the connected quads indices
                connected_quads_zero_crease_masked = arr_face_indices_zero_crease[connected_quads_zero_crease].reshape(
                    -1, 2)
                # calculating the vertex position for the edge points with zero crease
                edge_verts[idx_zero_crease] = self.midpoint_multiple(self.data['unique_edges'][idx_zero_crease][:, 0],
                                                                     self.data['unique_edges'][idx_zero_crease][:, 1],
                                                                     quad_ctr_list[
                                                                         connected_quads_zero_crease_masked[:, 0]],
                                                                     quad_ctr_list[
                                                                         connected_quads_zero_crease_masked[:, 1]])
                # array of the face indices repeated for every index in idx_crease
                array_face_indices_crease = np.tile(np.arange(len(self.data['faces'])), len(idx_crease))
                # boolean array where connected quads with crease are True; taken from faces_edge_matrix
                connected_quads = (self.data['faces_edges_matrix'][:, idx_crease] == 1).T.flatten()
                # masking the arrays to get only the connected quads indices
                connected_quads_masked = array_face_indices_crease[connected_quads].reshape(-1, 2)
                # calculating the smooth part of vertex position for the edge points with crease
                edge_points_smooth = self.midpoint_multiple(self.data['unique_edges'][idx_crease][:, 0],
                                                            # calculating the edge point coordinates
                                                            self.data['unique_edges'][idx_crease][:, 1],
                                                            quad_ctr_list[connected_quads_masked[:, 0]],
                                                            quad_ctr_list[connected_quads_masked[:, 1]])

                # calculating the sharp part of vertex position for the edge points with crease
                edge_points_sharp = self.midpoint_multiple(self.data['unique_edges'][idx_crease][:, 0],
                                                           self.data['unique_edges'][idx_crease][:, 1])

                # calculate edge points from smooth and sharp part
                edge_verts[idx_crease] = (
                    np.add(np.multiply((1 - self.data['creases'][idx_crease].reshape(-1, 1)), edge_points_smooth),
                           np.multiply(self.data['creases'][idx_crease].reshape(-1, 1), edge_points_sharp)))

            # calculating vertex position for edge points with infinite crease
            edge_verts[idx_infinite_crease] = self.midpoint_multiple(
                self.data['unique_edges'][idx_infinite_crease][:, 0],
                self.data['unique_edges'][idx_infinite_crease][:, 1])

        return edge_verts, idx_non_manifold_edges, idx_non_manifold_verts

    def refine(self):

        old_vertex_count = int(len(self.data['vertices']))

        #############################################################################################################
        # Part 1: Generating face points
        # -----------------------------------------------------------

        quad_ctr_list = self.generate_face_points()

        # Part 2: Generating edge points/ edge children
        # -----------------------------------------------------------

        edge_child_pos, idx_non_manifold_edges, idx_non_manifold_verts = self.generate_edge_points(quad_ctr_list)

        edges_with_crease = np.concatenate((self.data['unique_edges'], self.data['creases']),
                                           axis=1)  # append sharpness values to array of unique_edges

        self.data['vertices'] = np.concatenate((self.data['vertices'], edge_child_pos), axis=0)

        self._vertices_edges = edge_child_pos

        ################################################################################################################
        # Part 3: Updating the position of the old vertices
        # -----------------------------------------------------------

        m1 = self.data['vertices'][:old_vertex_count]

        verts_child = []
        sum_sharp_around_vertex = np.zeros(old_vertex_count)

        for i in range(old_vertex_count):

            connected_edge_mids = np.nonzero(self.data['vert_edge_list'].toarray()[i] == 1)[0]

            connected_verts = np.nonzero(self.data['vertices_adjacency_matrix'][i] == 1)[0]

            sharp_edges_around_vertex = np.count_nonzero(edges_with_crease[connected_edge_mids][:, 2] == 1,
                                                         keepdims=True)
            connected_quads_faces = (
                list(np.nonzero(np.array(self.data['faces']) == [i])[
                         0]))  # vertex connected to which face (face indices)

            edge_sharpness_idx = edges_with_crease[connected_edge_mids][:, 2]

            edges_sharpness_around_vertex = edge_sharpness_idx

            sum_sharp_around_vertex[i] = np.sum(edges_sharpness_around_vertex)

            number_faces = len(connected_quads_faces)
            number_edges = len(connected_verts)

            if i in idx_non_manifold_verts:

                if number_faces <= 3:
                    verts_child.append(m1[i])
                else:
                    print('vert:', i)
                    print(sharp_edges_around_vertex)
                    connected_non_manifold_edges = connected_edge_mids[
                        np.isin(connected_edge_mids, idx_non_manifold_edges)]

                    if len(connected_non_manifold_edges) + sharp_edges_around_vertex >= 3:
                        verts_child.append(m1[i])

                    else:

                        w1 = 0.75
                        connected_non_manifold_verts = np.unique(
                            self.data['unique_edges'][connected_non_manifold_edges])
                        w3 = np.where(np.isin(connected_verts, connected_non_manifold_verts), 0.125, 0)
                        m3 = self.midpoint_edges(connected_verts, w3)
                        verts_child.append(m1[i] * w1 + m3)

            elif sharp_edges_around_vertex < 2:

                if number_faces < 3 and number_edges <= 2:
                    w1 = 0.75
                    w3 = [0.125, 0.125]
                    m3 = self.midpoint_edges(connected_verts, w3)
                    verts_child.append(m1[i] * w1 + m3)

                elif number_faces < 3 and number_edges == 3:

                    if np.sum(self.data['vertices_adjacency_matrix'][connected_verts]) <= 7:
                        cVert = connected_verts[
                            np.nonzero(np.sum(self.data['vertices_adjacency_matrix'][connected_verts], axis=1) >= 3)]
                        connected_edges = self.data['unique_edges'][connected_edge_mids]
                        if i > cVert:
                            cEdge = connected_edge_mids[np.nonzero(connected_edges[:, 0] == cVert)[0]]
                        else:
                            cEdge = connected_edge_mids[np.nonzero(connected_edges[:, 1] == cVert)[0]]
                        pWeight = self.data['creases'][cEdge][0]
                        if pWeight > 0:
                            cWeight = 1.0 - pWeight
                            w1 = pWeight * 1.0 + cWeight * 0.75
                            w3 = np.where(np.sum(self.data['vertices_adjacency_matrix'][connected_verts], axis=1) < 3,
                                          cWeight * 0.125, 0)
                        else:
                            w3 = np.where(np.sum(self.data['vertices_adjacency_matrix'][connected_verts], axis=1) < 3,
                                          0.125, 0)
                            w1 = 0.75
                    else:
                        cVert = connected_verts[
                            np.nonzero(np.sum(self.data['vertex_faces_matrix'][connected_verts], axis=1) >= 3)]
                        connected_edges = self.data['unique_edges'][connected_edge_mids]
                        if i > cVert:
                            cEdge = connected_edge_mids[np.nonzero(connected_edges[:, 0] == cVert)[0]]
                        else:
                            cEdge = connected_edge_mids[np.nonzero(connected_edges[:, 1] == cVert)[0]]
                        pWeight = self.data['creases'][cEdge][0]
                        if pWeight > 0:
                            cWeight = 1.0 - pWeight
                            w1 = pWeight * 1.0 + cWeight * 0.75
                            w3 = np.where(np.sum(self.data['vertex_faces_matrix'][connected_verts], axis=1) < 3,
                                          cWeight * 0.125, 0)
                        else:
                            w3 = np.where(np.sum(self.data['vertex_faces_matrix'][connected_verts], axis=1) < 3, 0.125,
                                          0)
                            w1 = 0.75

                    m3 = self.midpoint_edges(connected_verts, w3)
                    verts_child.append(m1[i] * w1 + m3)
                else:
                    w1 = float((number_faces - 2) / number_faces)
                    count_zero_creases = np.count_nonzero(edges_sharpness_around_vertex == 0)
                    if count_zero_creases == 1:
                        pEdgeWeight = np.array([0.125, 0.125, 0.125])
                        idx_zero_sharpness = np.nonzero(edges_sharpness_around_vertex == 0)[0]
                        pEdgeWeight[idx_zero_sharpness] = 0

                        w2 = np.ones(number_faces) / (number_faces * number_faces)
                        w3 = w2

                        pWeight = self.fractional_weight(connected_edge_mids)
                        if pWeight > 0:
                            cWeight = 1.0 - pWeight
                            w1 = pWeight * 0.75 + cWeight * w1
                            w2 = cWeight * w2
                            w3 = pWeight * pEdgeWeight + cWeight * w3

                        m2 = self.midpoint_edges([quad_ctr_list[quadIdx] for quadIdx in connected_quads_faces], w2)

                        m3 = self.midpoint_edges(connected_verts, w3)
                        verts_child.append(m1[i] * w1 + m2 + m3)

                    elif count_zero_creases == 2:
                        m2 = self.midpoint(*[quad_ctr_list[quadIdx] for quadIdx in connected_quads_faces])
                        w2 = float(1 / number_faces)
                        w3 = np.ones(number_edges) / (number_faces * number_edges)
                        m3 = self.midpoint_edges(connected_verts, w3)
                        verts_child.append(m1[i] * w1 + m2 * w2 + m3)

                    else:
                        m2 = self.midpoint(*[quad_ctr_list[quadIdx] for quadIdx in connected_quads_faces])
                        w2 = float(1 / number_faces)
                        w3 = np.ones(number_edges) / (number_faces * number_edges)

                        if sum_sharp_around_vertex[i] == 0:
                            pWeight = 0
                        else:
                            pWeight = self.fractional_weight(connected_edge_mids)

                        if pWeight > 0:
                            cWeight = 1.0 - pWeight
                            w1 = pWeight * 1.0 + cWeight * w1
                            w2 = cWeight * w2
                            w3 = cWeight * w3

                        m3 = self.midpoint_edges(connected_verts, w3)
                        verts_child.append(m1[i] * w1 + m2 * w2 + m3)

                # n = float(len(connected_quads_faces))
                # w1 = float((n - 2) / n)
                # w2 = float(1 / n)
                # w3 = float(1 / n)

                # w1 = 0.75
                # w2 = 0
                # w3 = 0.25

                # verts_child.append(m1[i] * w1 + m2[i] * w2 + m3[i] * w3)



            elif sharp_edges_around_vertex > 2:

                verts_child.append(m1[i])

            else:

                pWeight = self.fractional_weight(connected_edge_mids)

                if 0 < pWeight < 1:
                    cWeight = 1.0 - pWeight
                    w1 = float((number_faces - 2) / number_faces)
                    w2 = float(1 / number_faces)
                    w3 = np.ones(number_edges) / (number_faces * number_edges)

                    w1 = pWeight * 1.0 + cWeight * w1
                    w2 = cWeight * w2
                    w3 = cWeight * w3
                    m2 = self.midpoint(*[quad_ctr_list[quadIdx] for quadIdx in connected_quads_faces])
                    m3 = self.midpoint_edges(connected_verts, w3)

                    verts_child.append(m1[i] * w1 + m2 * w2 + m3)
                else:

                    sorted_sharpness = []
                    for idx in connected_verts:

                        if idx < i:
                            sorted_sharpness.append(self.data['creases'][connected_edge_mids]
                                                    [edges_with_crease[connected_edge_mids][:, 0] == idx][0][0])
                        else:
                            sorted_sharpness.append(self.data['creases'][connected_edge_mids]
                                                    [edges_with_crease[connected_edge_mids][:, 1] == idx][0][0])

                    w1 = 0.75
                    w3 = np.where(np.array(sorted_sharpness) == 1, 0.125, 0)
                    m3 = self.midpoint_edges(connected_verts, w3)
                    verts_child.append(m1[i] * w1 + m3)

                    # print(number_edges)
                    # print(number_faces)
                    # for idx2 in second_edge[0]:
                    #    total_pos_forsharp = total_pos_forsharp + self.data['vertices'][i] + self.data['vertices'][
                    #        int(idx2)]
                    # print(total_pos_forsharp)
                    # print(m1[i])
                    # verts_child.append((total_pos_forsharp + 4 * m1[i]) / 8)

                # for idx2 in connected_verts:
                # total_pos_forsharp = total_pos_forsharp + self.data['vertices'][i] + self.data['vertices'][
                # int(idx2)]
                # print("total_pos_forsharp", total_pos_forsharp)
                # v_s = 0
                # for esAv in edges_sharpness_around_vertex:
                # v_s = v_s + esAv
                # v_s = v_s / len(edges_sharpness_around_vertex)
                # print("v_s", v_s)
                # if v_s >= 1:
                # verts_child.append((total_pos_forsharp + 4 * m1[i]) / 8)
                # print((total_pos_forsharp + 4 * m1[i]) / 8)
                # else:
                # print((total_pos_forsharp + 4 * m1[i]) / 8)
                # verts_child.append((total_pos_forsharp + 4 * m1[i]) / 8)

        self.data['vertices'][:old_vertex_count] = verts_child

        ###############################################################################################################
        # Part 4: Create subdivided model
        # -----------------------------------------------------------

        mat = (np.zeros((4 * len(self.data['faces']), 4), dtype=int))
        mat[:, 0] = np.repeat(quad_ctr_list, 4)
        mat[:, 2] = np.ravel(self.data['faces'])
        indices = np.arange(len(self.data['unique_edges']))
        idx_edge_points = indices + np.max(quad_ctr_list) + 1
        edges_mid = np.concatenate((self.data['unique_edges'], idx_edge_points.reshape(len(idx_edge_points), 1)),
                                   axis=1)
        mpoint1 = []
        mpoint2 = []

        for curQuad, quadVerts in enumerate(self.data['faces'], 0):
            poly_count = len(quadVerts)

            for quadIdx in range(len(quadVerts)):
                idx0 = quadVerts[(quadIdx - 1) % poly_count]
                idx1 = quadVerts[quadIdx]
                idx2 = quadVerts[(quadIdx + 1) % poly_count]

                if idx0 < idx1:
                    mpoint1.append(edges_mid[(edges_mid[:, 0] == idx0) & (edges_mid[:, 1] == idx1)][:, 2])

                else:
                    mpoint1.append(edges_mid[(edges_mid[:, 0] == idx1) & (edges_mid[:, 1] == idx0)][:, 2])

                if idx1 < idx2:
                    mpoint2.append(edges_mid[(edges_mid[:, 0] == idx1) & (edges_mid[:, 1] == idx2)][:, 2])

                else:
                    mpoint2.append(edges_mid[(edges_mid[:, 0] == idx2) & (edges_mid[:, 1] == idx1)][:, 2])

        mat[:, 1] = np.hstack(mpoint1)
        mat[:, 3] = np.hstack(mpoint2)

        to_edges = np.sort(calculation.faces_to_edges(mat, self.data['mesh_type']))

        new_edges, indices = np.unique(to_edges, return_index=True, axis=0)
        new_edges = np.vstack([to_edges[index] for index in sorted(indices)])

        crease_around_vertex = np.zeros((len(new_edges), 1))
        mid_edge_crease = np.zeros((len(new_edges), 1))

        idx_vert_with_crease = np.nonzero(sum_sharp_around_vertex > 0)[0]

        for idx in idx_vert_with_crease:
            arr_idx = np.nonzero(new_edges[:, 0] == idx)
            crease_around_vertex[arr_idx] = sum_sharp_around_vertex[idx]

        edges_mid_crease = np.hstack((edges_mid, self.data['creases']))

        mid_point_crease = edges_mid_crease[edges_mid_crease[:, 3] > 0][:, 2:]

        new_edges_masked = []

        for i in range(len(mid_point_crease)):
            new_edges_masked.append(
                np.nonzero((new_edges[:, 0] < old_vertex_count) & (new_edges[:, 1] == mid_point_crease[i, 0])))
            mid_edge_crease[new_edges_masked[i]] = mid_point_crease[i, 1]

        new_edges_crease = np.where((mid_edge_crease == 1) | (crease_around_vertex - mid_edge_crease == 1), 1,
                                    np.where(crease_around_vertex * 0.25 + mid_edge_crease * 0.5 - 1 > 0,
                                             crease_around_vertex * 0.25 + mid_edge_crease * 0.5 - 1, 0))

        self.data['vertices'] = self.data['vertices'].tolist()

        self.data['faces'] = mat
        edges = calculation.faces_to_edges(self.data['faces'], self.data['mesh_type'])
        self.data['unique_edges'] = new_edges
        self.edges_list.append(new_edges)
        self.data['creases'] = new_edges_crease
        self.creases_list.append(new_edges_crease)
        self.data['faces_edges_matrix'] = data_structure.face_edges_incident(self.data['unique_edges'], edges,
                                                                             self.data['faces'], self.data['mesh_type'])

        graph = data_structure.mesh_graph(self.data['vertices'], self.data['unique_edges'])
        self.data['vert_edge_list'] = data_structure.vert_edge_incident(graph, self.data['unique_edges'],
                                                                        self.data['vertices'])
        self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(graph, self.data['vertices'],
                                                                                  matrix_form=True)
        self.data['vertex_faces_matrix'] = data_structure.vertex_face_incident(self.data['faces'],
                                                                               self.data['vertices'])
