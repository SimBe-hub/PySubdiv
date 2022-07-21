import numpy as np
from PySubdiv.backend import calculation
from PySubdiv.data import data_structure, data
from PySubdiv.subdivision_algorithms.subdivision_backend import vertex_masks
from PySubdiv.subdivision_algorithms.subdivision_backend import property_inhertiage
from scipy.sparse import lil_matrix


class MeshRefiner(object):
    # initialize the necessary data from the mesh
    def __init__(self, mesh):
        self.data = data.data_dictionary()
        self.data['vertices'] = mesh.data["vertices"]
        self.data['faces'] = mesh.data["faces"]
        self.data['mesh_type'] = mesh.data['mesh_type']

        if 'unique_edges' in mesh.data:
            self.data['unique_edges'] = mesh.data["unique_edges"]
            self.data['edges'] = mesh.data['edges']
        else:
            self.data['unique_edges'] = mesh.edges_unique()
            self.data['edges'] = mesh.data['edges']

        if 'creases' in mesh.data:
            self.data['creases'] = mesh.data['creases']
        else:
            self.data['creases'] = mesh.set_crease()

        if 'edge_faces_dictionary' in mesh.data:
            self.data['edge_faces_dictionary'] = mesh.data['edge_faces_dictionary']
        else:
            self.data['edge_faces_dictionary'] = mesh.edges_faces_connected()

        if 'vertex_edges_dictionary' in mesh.data:
            self.data['vertex_edges_dictionary'] = mesh.data['vertex_edges_dictionary']
        else:
            self.data['vertex_edges_dictionary'] = mesh.vertex_edges_dictionary()

        if 'vertices_connected_dictionary' in mesh.data:
            self.data['vertices_connected_dictionary'] = mesh.data['vertices_connected_dictionary']
        else:
            self.data['vertices_connected_dictionary'] = mesh.vertices_connected_dictionary()
        if 'vertex_faces_dictionary' in mesh.data:
            self.data['vertex_faces_dictionary'] = mesh.data['vertex_faces_dictionary']
        else:
            self.data['vertex_faces_dictionary'] = mesh.vertex_faces_dictionary()
        if 'dynamic_faces' in mesh.data:
            self.data['dynamic_faces'] = mesh.data['dynamic_faces']
        if 'dynamic_vertices' in mesh.data:
            self.data['dynamic_vertices'] = mesh.data['dynamic_vertices']
        if 'volumes' in mesh.data:
            self.data['volumes'] = mesh.data['volumes']
            self._inherit_volumes = True
        else:
            self._inherit_volumes = False

        if 'boundary_vertices' in mesh.data:
            self.data['boundary_vertices'] = mesh.data['boundary_vertices']

        if 'non_manifold_vertices' in mesh.data:
            self.data['non_manifold_vertices'] = mesh.data['non_manifold_vertices']

        if 'boundary_layer_vertices' in mesh.data:
            self.data['boundary_layer_vertices'] = mesh.data['boundary_layer_vertices']
        if 'boundary_edges_data' in mesh.data:
            self.data['boundary_edges_data'] = mesh.data['boundary_edges_data']

        if 'subdiv_iterations' in mesh.data:
            self.data['subdiv_iterations'] = mesh.data['subdiv_iterations']
        else:
            self.data['subdiv_iterations'] = 0
        if 'fitting_method' in mesh.data:
            self.data['fitting_method'] = mesh.data['fitting_method']
        else:
            self.data['fitting_method'] = None

    def add_vertex(self, v):
        self.data['vertices'].append(v)
        return len(self.data['vertices']) - 1

    # def add_vertex_multiple(self, v):
    #     start = len(self.data['vertices'])
    #     self.data['vertices'].extend(v)
    #     return np.arange(start, len(self.data['vertices']))

    # def quads_contain(self, *indices):
    #     connected = []
    #
    #     # For every quad
    #     for curQuad, quadVerts in enumerate(self.data['faces'], 0):
    #
    #         # If every indice exists in the quad, add it to our list
    #         for idx in indices:
    #             if idx not in quadVerts:
    #                 break
    #         else:
    #             connected.append(curQuad)
    #
    #     return connected

    # def midpoint(self, *indices):
    #     mpoint = np.array([0, 0, 0])
    #     for idx in indices:
    #         mpoint = mpoint + self.data['vertices'][idx]
    #     mpoint = mpoint / (len(indices))
    #     return mpoint

    # def midpoint_multiple(self, idx_arr0, idx_arr1, idx_arr2=None, idx_arr3=None):
    #
    #     vertices = np.asanyarray(self.data['vertices'])
    #     if idx_arr2 is None and idx_arr3 is None:
    #         mpoint = (np.add(vertices[[idx_arr0]], vertices[[idx_arr1]])) / 2
    #     else:
    #         if self.data['mesh_type'] == 'quadrilateral':
    #             add_1 = np.add(vertices[[idx_arr0]], vertices[[idx_arr1]])
    #             add_2 = np.add(vertices[[idx_arr2]], vertices[[idx_arr3]])
    #             mpoint = np.add(add_1, add_2) / 4
    #         else:
    #             add_1 = np.add(vertices[[idx_arr0]], vertices[[idx_arr1]]) * 0.375
    #             add_2 = np.add(vertices[[idx_arr2]], vertices[[idx_arr3]]) * 0.125
    #             mpoint = np.add(add_1, add_2)
    #
    #     return mpoint

    # def midpoint_edges_old(self, *indices):
    #     mpoint = np.array([0, 0, 0])
    #     for idx in indices:
    #         mpoint = mpoint + self.data['vertices'][idx]
    #     mpoint = mpoint / (len(indices))
    #     return mpoint

    # def midpoint_edges(self, indices, weights):
    #     mpoint = np.sum(self.data['vertices'][indices] * weights.reshape(-1, 1), axis=0)
    #     return mpoint

    # def fractional_weight(self, index_incident_edges):
    #     transition_count = 0
    #     transition_sum = 0
    #     for idx in index_incident_edges:
    #         transition_sum += self.data['creases'][idx]
    #         if self.data['creases'][idx] > 0:
    #             transition_count += 1
    #
    #     fractional_weight = transition_sum / transition_count
    #     return fractional_weight

    # def generate_face_points(self):
    #     faces_arr = np.asanyarray(self.data['faces'])  # convert list of faces to np.array
    #     verts_arr = np.asanyarray(self.data['vertices'])  # convert list of vertices to np.array
    #     # calculate coordinates of face points
    #     face_points = (np.sum(verts_arr[faces_arr], axis=1) / len(faces_arr[0])).tolist()
    #
    #     quad_ctr_list = self.add_vertex_multiple(face_points)  # List of indices of the facepoints in the vertices array
    #     return quad_ctr_list

    def generate_edge_points(self, subdivison_weight_matrix):
        # np.array to store vertices of the edge points
        edge_verts = np.zeros((len(self.data['unique_edges']), 3))
        old_vertex_count = len(self.data['vertices'])

        # seperate boundary odd edges and non manifold edges from edge vertices

        idx_boundary_odd_edges = []
        idx_non_manifold_edges = []
        idx_edges = []
        for edge_index in self.data['edge_faces_dictionary']:
            number_connected_faces = len(self.data['edge_faces_dictionary'][edge_index])
            if number_connected_faces <= 1:
                idx_boundary_odd_edges.append(edge_index)
            elif number_connected_faces >= 3:
                idx_non_manifold_edges.append(edge_index)
            else:
                idx_edges.append(edge_index)

        idx_boundary_odd_edges = np.array(idx_boundary_odd_edges)
        idx_non_manifold_edges = np.array(idx_non_manifold_edges)
        idx_edges = np.array(idx_edges)

        if len(idx_non_manifold_edges) == 0:
            idx_non_manifold_verts = np.array([])
        else:
            idx_non_manifold_verts = np.unique(self.data['unique_edges'][idx_non_manifold_edges])

        if len(idx_edges != 0):
            idx_zero_crease = idx_edges[np.nonzero(self.data['creases'][idx_edges] == 0)[0]]
            # array with indices of unique_edges where crease == 1
            idx_infinite_crease = idx_edges[np.nonzero(self.data['creases'][idx_edges] == 1)[0]]

            # array with indices of unique_edges where crease > 0 & < 1
            idx_crease = idx_edges[np.nonzero((self.data['creases'][idx_edges] > 0)
                                              & (self.data['creases'][idx_edges] < 1))[0]]
        else:
            idx_zero_crease = []
            idx_infinite_crease = []
            idx_crease = []

        # calculating the position for boundary odd vertices with zero crease #part must be revisited
        # creases for boundary odd vertices might  not work correctly
        if len(idx_boundary_odd_edges != 0):
            for idx in idx_boundary_odd_edges:
                subdivison_weight_matrix[idx + old_vertex_count, self.data['unique_edges'][idx]] = 0.5

        # can be put together with boundary odd vertices to shorten the code
        if len(idx_non_manifold_edges != 0):
            for idx in idx_non_manifold_edges:
                subdivison_weight_matrix[idx + old_vertex_count, self.data['unique_edges'][idx]] = 0.5

        if len(idx_zero_crease) > 0:

            # array of the face indices repeated for every index in idx_zero_crease
            arr_face_indices_zero_crease = np.tile(np.arange(len(self.data['faces'])), len(idx_zero_crease))

            # boolean array where connected tris with zero crease are True; taken from faces_edge_matrix
            connected_tris_zero_crease = np.zeros((len(idx_zero_crease), 2), dtype=int)
            for counter, index in enumerate(idx_zero_crease):
                connected_tris_zero_crease[counter] = self.data['edge_faces_dictionary'][index]

            # masking the arrays to get only the connected triangular indices
            connected_tris_zero_crease_masked = arr_face_indices_zero_crease[connected_tris_zero_crease].reshape(
                -1, 2)
            # finding vertices opposite to the new edge point
            opposite_verts_zero_crease = np.zeros((len(connected_tris_zero_crease_masked), 2), dtype=int)
            for i in range(len(connected_tris_zero_crease_masked)):
                connected_face_1 = set(self.data['faces'][connected_tris_zero_crease_masked[i][0]])
                connected_face_2 = set(self.data['faces'][connected_tris_zero_crease_masked[i][1]])
                opposite_verts_zero_crease[i, 0] = list(connected_face_1 - connected_face_2)[0]
                opposite_verts_zero_crease[i, 1] = list(connected_face_2 - connected_face_1)[0]

            # calculating the vertex position for the edge points with zero crease
            for counter, idx in enumerate(idx_zero_crease):
                subdivison_weight_matrix[idx + old_vertex_count, self.data['unique_edges'][idx]] = 0.375
                subdivison_weight_matrix[idx + old_vertex_count, opposite_verts_zero_crease[counter]] = 0.125

        if len(idx_crease) > 0:
            # array of the face indices repeated for every index in idx_crease
            array_face_indices_crease = np.tile(np.arange(len(self.data['faces'])), len(idx_crease))
            # boolean array where connected tris with crease are True; taken from faces_edge_matrix
            connected_tris = np.zeros((len(idx_crease), 2), dtype=int)
            for counter, index in enumerate(idx_crease):
                connected_tris[counter] = self.data['edge_faces_dictionary'][index]
            # masking the arrays to get only the connected tris indices
            connected_tris_masked = array_face_indices_crease[connected_tris].reshape(-1, 2)
            # finding vertices opposite to the new edge point
            opposite_verts = np.zeros((len(connected_tris_masked), 2), dtype=int)
            for i in range(len(connected_tris_masked)):
                connected_face_1 = set(self.data['faces'][connected_tris_masked[i][0]])
                connected_face_2 = set(self.data['faces'][connected_tris_masked[i][1]])
                opposite_verts[i, 0] = list(connected_face_1 - connected_face_2)[0]
                opposite_verts[i, 1] = list(connected_face_2 - connected_face_1)[0]

            edge_weights_smooth = (1 - self.data['creases'][idx_crease].reshape(-1, 1)) * \
                                  np.array([0.375, 0.375, 0.125, 0.125])  # smooth edge weights
            edge_weights_sharp = (self.data['creases'][idx_crease].reshape(-1, 1)) * \
                                 np.array([0.5, 0.5, 0, 0])  # sharp edge weights
            vertex_weight_combined = edge_weights_smooth + edge_weights_sharp
            for counter, idx in enumerate(idx_crease):
                subdivison_weight_matrix[idx + old_vertex_count, self.data['unique_edges'][idx]] = \
                    vertex_weight_combined[counter][0]
                subdivison_weight_matrix[idx + old_vertex_count, opposite_verts[counter]] = \
                    vertex_weight_combined[counter][2]

        if len(idx_infinite_crease) > 0:
            for idx in idx_infinite_crease:
                subdivison_weight_matrix[idx + old_vertex_count, self.data['unique_edges'][idx]] = 0.5

        return edge_verts, idx_boundary_odd_edges, idx_non_manifold_edges, idx_non_manifold_verts, \
               idx_infinite_crease, idx_crease, subdivison_weight_matrix

    def refine(self):
        old_vertex_count = int(len(self.data['vertices']))
        subdivision_weight_matrix = lil_matrix((old_vertex_count + len(self.data['unique_edges']), old_vertex_count))

        #############################################################################################################
        # # Part 1: Generating face points
        # # -----------------------------------------------------------
        # if self.data['mesh_type'] == 'quadrilateral':
        #     quad_ctr_list = self.generate_face_points()
        # else:
        #     # triangular meshes don't require face points
        #     pass

        # Part 2: Generating edge points/ edge children
        # -----------------------------------------------------------
        edge_child_pos, idx_boundary_odd_edges, idx_non_manifold_edges, idx_non_manifold_verts, \
        idx_infinite_crease_edges, idx_crease_edges, subdivision_weight_matrix = self.generate_edge_points(
            subdivision_weight_matrix)
        subdivision_weight_matrix = subdivision_weight_matrix
        if len(idx_boundary_odd_edges != 0):
            idx_boundary_verts = np.unique(self.data['unique_edges'][idx_boundary_odd_edges])
        else:
            idx_boundary_verts = np.array([])

        edges_with_crease = np.concatenate((self.data['unique_edges'], self.data['creases']),
                                           axis=1)  # append sharpness values to array of unique_edges

        # set crease value of non-manifold edges to zero as they don't influence the calculation
        if len(idx_non_manifold_edges) > 0:
            edges_with_crease[idx_non_manifold_edges, 2] = 0

        edge_sharpness_idx = np.nonzero(edges_with_crease[:, 2])

        self.data['vertices'] = np.concatenate((self.data['vertices'], edge_child_pos), axis=0)

        self._vertices_edges = edge_child_pos

        ################################################################################################################
        # Part 3: Updating the position of the old vertices
        # -----------------------------------------------------------
        sum_sharp_around_vertex = np.zeros(old_vertex_count)
        vertex_edges_connected = self.data['vertex_edges_dictionary']
        for i in range(old_vertex_count):

            connected_verts = self.data['vertices_connected_dictionary'][i]
            connected_edge_mids = vertex_edges_connected[i]
            sharp_edges_around_vertex = np.count_nonzero(edges_with_crease[connected_edge_mids][:, 2] == 1,
                                                         keepdims=True)

            connected_faces = self.data['vertex_faces_dictionary'][i]
            edges_sharpness_around_vertex = edges_with_crease[connected_edge_mids][:, 2]

            number_creases_around_vertex = np.count_nonzero((edges_sharpness_around_vertex > 0) &
                                                            (edges_sharpness_around_vertex < 1))

            sum_sharp_around_vertex[i] = np.sum(edges_sharpness_around_vertex)
            number_faces = len(connected_faces)
            number_edges = len(connected_verts)

            if i in idx_non_manifold_verts:
                connected_non_manifold_edges = connected_edge_mids[
                    np.isin(connected_edge_mids, idx_non_manifold_edges)]
                # if number_faces <= 3 or i in idx_boundary_verts:
                # if only one non-manifold edge connected -> complex non-manifold vertex -> corner rule
                if len(connected_non_manifold_edges) == 1:
                    subdivision_weight_matrix[i, i] = 1  # sparse matrix
                else:
                    # if more than two non-manifold edge connected -> complex non-manifold vertex -> corner rule
                    if len(connected_non_manifold_edges) > 2:
                        subdivision_weight_matrix[i, i] = 1  # sparse matrix
                    # two non-manifold-vertices connected crease-rule
                    # non-manifold vertices only influenced by adjacent non-manifold vertices
                    else:
                        # find idx of the connected non-manifold vertices
                        connected_non_manifold_verts = np.unique(
                            self.data['unique_edges'][connected_non_manifold_edges])
                        if sum_sharp_around_vertex[i] == 0:
                            # no csv assigned to adjacent edges (only manifold edges count) -> crease-rule
                            # 0.75 on original vertex, 0.125 on the two connected non-manifold vertices
                            w1 = 0.75
                            w3 = np.where(np.isin(connected_verts, connected_non_manifold_verts), 0.125, 0)
                            subdivision_weight_matrix[i, i] = w1
                            subdivision_weight_matrix[i, connected_verts] = w3
                        else:
                            # csv assigned to adjacent edges (only manifold edges count)
                            # -> corner rule vertex and crease rule on the two connected non-manifold vertices
                            # fractional weight must be calculated

                            # corner mask
                            w1_corner = 1.0
                            w3_corner = 0

                            # crease mask
                            w1_sharp = 0.75
                            w3_sharp = np.where(np.isin(connected_verts, connected_non_manifold_verts), 0.125, 0)
                            # fractional weight
                            pWeight = vertex_masks.fractional_weight(self, connected_edge_mids)
                            cWeight = 1.0 - pWeight
                            w1_combined, w3_combined = vertex_masks.combine_vertex_masks(w1_corner, w1_sharp, w3_corner,
                                                                                         w3_sharp,
                                                                                         pWeight, cWeight)

                            subdivision_weight_matrix[i, i] = w1_combined
                            subdivision_weight_matrix[i, connected_verts] = w3_combined

            elif i in idx_boundary_verts:
                # parent vertices are boundary vertices

                if number_creases_around_vertex + sharp_edges_around_vertex >= 2:

                    # corner mask for parent vertex, smooth mask for child vertices
                    w1_sharp = 1.0
                    w3_sharp = 0
                    w1_smooth, w3_smooth = vertex_masks.calculate_smooth_mask_for_vertex(number_faces, number_edges)

                    pWeight = vertex_masks.fractional_weight(self, connected_edge_mids)
                    cWeight = 1.0 - pWeight
                    w1_combined, w3_combined = vertex_masks.combine_vertex_masks(w1_sharp, w1_smooth, w3_sharp,
                                                                                 w3_smooth,
                                                                                 pWeight, cWeight)
                    subdivision_weight_matrix[i, i] = w1_combined
                    subdivision_weight_matrix[i, connected_verts] = w3_combined

                else:

                    w1 = 0.75
                    connected_crease_edges = connected_edge_mids[
                        np.isin(connected_edge_mids, idx_boundary_odd_edges)]
                    connected_crease_verts = np.unique(self.data['unique_edges'][connected_crease_edges])
                    w3 = np.where(np.isin(connected_verts, connected_crease_verts), 0.125, 0)
                    subdivision_weight_matrix[i, i] = w1
                    subdivision_weight_matrix[i, connected_verts] = w3

            # parent and boundary vertices are smooth when no incident edge is sharp/crease
            # a vertex with one sharp edge/crease is a dart and calculated with the smooth vertex rule

            elif sum_sharp_around_vertex[i] == 0 or (number_creases_around_vertex + sharp_edges_around_vertex == 1):

                # parent and child vertices are smooth
                w1_smooth, w3_smooth = vertex_masks.calculate_smooth_mask_for_vertex(number_faces, number_edges)
                subdivision_weight_matrix[i, i] = w1_smooth
                subdivision_weight_matrix[i, connected_verts] = w3_smooth
            else:
                if number_creases_around_vertex + sharp_edges_around_vertex > 2:

                    # corner mask for parent vertex, smooth mask for child vertices
                    w1_sharp = 1.0
                    w3_sharp = 0
                    w1_smooth, w3_smooth = vertex_masks.calculate_smooth_mask_for_vertex(number_faces, number_edges)

                else:
                    # crease mask for parent vertex, smooth mask for child vertices
                    crease_edges = np.concatenate((idx_infinite_crease_edges, idx_crease_edges))
                    w1_sharp, w3_sharp = vertex_masks.calculate_crease_mask_for_vertex(self, connected_edge_mids,
                                                                                       connected_verts,
                                                                                       crease_edges)
                    w1_smooth, w3_smooth = vertex_masks.calculate_smooth_mask_for_vertex(number_faces, number_edges)
                pWeight = vertex_masks.fractional_weight(self, connected_edge_mids)
                cWeight = 1.0 - pWeight
                w1_combined, w3_combined = vertex_masks.combine_vertex_masks(w1_sharp, w1_smooth, w3_sharp, w3_smooth,
                                                                             pWeight, cWeight)
                subdivision_weight_matrix[i, i] = w1_combined
                subdivision_weight_matrix[i, connected_verts] = w3_combined


        subdivision_weight_matrix = subdivision_weight_matrix.tocsr()
        self.data['subdivision_weight_matrix'] = subdivision_weight_matrix
        self.data['vertices'] = subdivision_weight_matrix.dot(self.data['vertices'][:old_vertex_count])

        ###############################################################################################################
        # Part 4: Build the topology for the new mesh
        # -----------------------------------------------------------
        # 4 child faces per parent face with 3 vertices position.
        new_faces = np.zeros((4 * len(self.data['faces']), len(self.data['faces'][0])), dtype=int)

        # find indices of the new edge points
        idx_edge_mids = np.arange(len(self.data['unique_edges']))

        first_idx_edge_points = idx_edge_mids + old_vertex_count

        edges_directed = np.zeros((len(self.data['unique_edges']) * 2, 3), dtype=int)

        rolled = np.roll(self.data['unique_edges'], 1, axis=1)

        edges_mid_direction_1 = np.concatenate(
            (self.data['unique_edges'], first_idx_edge_points.reshape(len(first_idx_edge_points), 1)),
            axis=1)

        edges_mid_direction_2 = np.concatenate(
            (rolled, first_idx_edge_points.reshape(len(first_idx_edge_points), 1)),
            axis=1)

        edges_directed[::2] = edges_mid_direction_1
        edges_directed[1::2] = edges_mid_direction_2

        midpoint_edges_dict = {tuple(edge_mid[:2]): edge_mid[2] for edge_mid in edges_directed}
        midpoint_edges = []
        for edge in self.data["edges"]:
            midpoint_edges.append(midpoint_edges_dict[tuple(edge)])

        np.fill_diagonal(new_faces, self.data['faces'], wrap=True)

        new_faces = np.roll(new_faces, 1, axis=1)

        new_faces[3::4] = np.reshape(midpoint_edges, (len(new_faces[3::4]), 3))
        np.fill_diagonal(new_faces, np.roll(np.reshape(midpoint_edges, (-1, 3)), 1, axis=1).flatten(), wrap=True)
        new_faces = np.roll(new_faces, 1, axis=1)
        np.fill_diagonal(new_faces, midpoint_edges, wrap=True)
        new_faces = np.roll(new_faces, 1, axis=1)

        to_edges = np.sort(calculation.faces_to_edges(new_faces, self.data['mesh_type']))

        new_edges, indices = np.unique(to_edges, return_index=True, axis=0)
        new_edges = np.vstack([to_edges[index] for index in sorted(indices)])


        ###############################################################################################################
        # Part 5: Pass down data to the refinement
        # -----------------------------------------------------------

        # calculate CSV for the new edges. That information is derived from edges and vertices of the coarse mesh.
        # arrays to store sum of CSV around each edge and CSV of edge itself


        crease_around_vertex = np.zeros((len(new_edges), 1))
        mid_edge_crease = np.zeros((len(new_edges), 1))
        # find old vertices with that are adjacent to old edges with crease
        idx_vert_with_crease = np.nonzero(sum_sharp_around_vertex > 0)[0]
        # find crease value of the new
        for idx in idx_vert_with_crease:
            # find edges that are connected to the crease vertex
            arr_idx = np.nonzero(new_edges[:, 0] == idx)[0]
            crease_around_vertex[arr_idx] = sum_sharp_around_vertex[idx]
        edges_mid_crease = np.hstack((edges_mid_direction_1, self.data['creases']))

        mid_point_crease = edges_mid_crease[edges_mid_crease[:, 3] > 0][:, 2:]

        for i in range(len(mid_point_crease)):
            new_edges_masked = np.nonzero(
                (new_edges[:, 0] < old_vertex_count) & (new_edges[:, 1] == mid_point_crease[i, 0]))
            mid_edge_crease[new_edges_masked] = mid_point_crease[i, 1]


        # can yield crease values > 1 must be looked into there might be an error in calculation
        new_edges_crease = np.where((mid_edge_crease == 1) | (crease_around_vertex - mid_edge_crease == 1), 1,
                                    np.where(crease_around_vertex * 0.25 + mid_edge_crease * 0.75 - 1 > 0,
                                             crease_around_vertex * 0.25 + mid_edge_crease * 0.75 - 1, 0))
        # correct crease values > 1 by setting them to 1
        new_edges_crease = np.where(new_edges_crease > 1, 1, new_edges_crease)

        # set dynamic vertices for fitting purposes with signed distance function
        if 'boundary_layer_vertices' in self.data:
            self.data['boundary_layer_vertices'] = \
                property_inhertiage.pass_down_layer_boundary_vertices(edges_mid_direction_1,
                                                                      self.data['boundary_layer_vertices'],
                                                                      self.data['dynamic_vertices'],
                                                                      self.data['edge_faces_dictionary'],
                                                                      self.data['dynamic_faces'])

        if "dynamic_faces" in self.data and "dynamic_vertices" in self.data:
            dynamic_faces_vertices_edges = property_inhertiage.pass_down_dynamic_faces_vertices(
                self.data['dynamic_faces'], self.data['dynamic_vertices'], self.data['unique_edges'],
                self.data['edge_faces_dictionary'], 'dynamic_faces')

            self.data['dynamic_faces'] = dynamic_faces_vertices_edges[0]
            self.data['dynamic_vertices'] = dynamic_faces_vertices_edges[1]
            self.data['dynamic_edges'] = dynamic_faces_vertices_edges[2]

        # inherit volumes for subdivided faces:
        if self._inherit_volumes:
            volumes_inherited = property_inhertiage.pass_down_volumes(self.data['volumes'])
            self.data['volumes'] = volumes_inherited

        if 'boundary_vertices' in self.data:
            self.data['boundary_vertices'] = \
                property_inhertiage.pass_down_boundary_vertices(edges_mid_direction_1,
                                                                self.data['boundary_vertices'])

        if 'non_manifold_vertices' in self.data:
            self.data['non_manifold_vertices'] = \
                property_inhertiage.pass_down_boundary_vertices(edges_mid_direction_1,
                                                                self.data['non_manifold_vertices'])
        if 'boundary_edges_data' in self.data:
            self.data['boundary_edges_data'] = \
                property_inhertiage.pass_down_boundary_edge_data(self.data['edges'], new_edges, edges_mid_direction_1,
                                                                 self.data['boundary_edges_data'])


        self.data['faces'] = new_faces
        edges = calculation.faces_to_edges(self.data['faces'], self.data['mesh_type'])
        self.data['edges'] = edges
        self.data['unique_edges'] = new_edges
        self.data['creases'] = new_edges_crease

        self.data['edge_faces_dictionary'] = data_structure.edge_faces_dict(self.data['unique_edges'], edges)
        self.data['vertex_edges_dictionary'] = data_structure.vertex_edges_dict(self.data['vertices'],
                                                                                self.data['unique_edges'])

        self.data['vertices_connected_dictionary'] = data_structure.vertex_vertex_dictionary(self.data['vertices'],
                                                                                             self.data['unique_edges'])
        self.data['vertex_faces_dictionary'] = data_structure.vertex_faces_dict(self.data['faces'],
                                                                                self.data['vertices'])

