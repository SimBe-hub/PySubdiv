import numpy as np
from pysubdiv.backend import calculation
from pysubdiv.main.data import data_structure
from pysubdiv.main.data import data
from pysubdiv.main.subdivision_algorithms.subdivision_backend import vertex_masks
from pysubdiv.main.subdivision_algorithms.subdivision_backend import property_inhertiage


class Structure(object):

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

        if 'dynamic_faces' in mesh.data:
            self.data['dynamic_faces'] = mesh.data['dynamic_faces']

        if 'dynamic_vertices' in mesh.data:
            self.data['dynamic_vertices'] = mesh.data['dynamic_vertices']

        if 'volumes' in mesh.data:
            self.data['volumes'] = mesh.data['volumes']
            self._inherit_volumes = True
        else:
            self._inherit_volumes = False

        if 'subdiv_iterations' in mesh.data:
            self.data['subdiv_iterations'] = mesh.data['subdiv_iterations']
        else:
            self.data['subdiv_iterations'] = 0

        if 'fitting_method' in mesh.data:
            self.data['fitting_method'] = mesh.data['fitting_method']
        else:
            self.data['fitting_method'] = None

        if mesh.data['interactive']:
            self._interactive = True
            self.data['vertices'][:len(self.data['ctr_points'])] = self.data['ctr_points']

        else:
            self._interactive = False

        self._vertices_edges = []

    def add_vertex(self, v):
        self.data['vertices'].append(v)
        return len(self.data['vertices']) - 1

    def add_vertex_multiple(self, v):
        start = len(self.data['vertices'])
        self.data['vertices'].extend(v)
        return np.arange(start, len(self.data['vertices']))

    def quads_contain(self, *indices):
        connected = []

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
            if self.data['mesh_type'] == 'quadrilateral':
                add_1 = np.add(vertices[[idx_arr0]], vertices[[idx_arr1]])
                add_2 = np.add(vertices[[idx_arr2]], vertices[[idx_arr3]])
                mpoint = np.add(add_1, add_2) / 4
            else:
                add_1 = np.add(vertices[[idx_arr0]], vertices[[idx_arr1]]) * 0.375
                add_2 = np.add(vertices[[idx_arr2]], vertices[[idx_arr3]]) * 0.125
                mpoint = np.add(add_1, add_2)

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

    def generate_edge_points(self):
        # np.array to store vertices of the edge points
        edge_verts = np.zeros((len(self.data['unique_edges']), 3))

        # seperate boundary odd edges and non manifold edges from edge vertices

        idx_boundary_odd_edges = np.nonzero(np.sum(self.data['faces_edges_matrix'], axis=0) <= 1)[0]
        idx_non_manifold_edges = np.nonzero(np.sum(self.data['faces_edges_matrix'], axis=0) >= 3)[0]
        idx_non_manifold_verts = np.unique(self.data['unique_edges'][idx_non_manifold_edges])

        idx_edges = np.nonzero((np.sum(self.data['faces_edges_matrix'], axis=0) > 1) &
                               (np.sum(self.data['faces_edges_matrix'], axis=0) < 3))[0]

        idx_zero_crease = idx_edges[np.nonzero(self.data['creases'][idx_edges] == 0)[0]]
        # array with indices of unique_edges where crease == 1
        idx_infinite_crease = idx_edges[np.nonzero(self.data['creases'][idx_edges] == 1)[0]]

        # array with indices of unique_edges where crease > 0 & < 1
        idx_crease = idx_edges[np.nonzero((self.data['creases'][idx_edges] > 0)
                                          & (self.data['creases'][idx_edges] < 1))[0]]

        # calculating the position for boundary odd vertices with zero crease #part must be revisited
        # creases for boundary odd vertices might  not work correctly
        edge_verts[idx_boundary_odd_edges] = self.midpoint_multiple(
            self.data['unique_edges'][idx_boundary_odd_edges][:, 0],
            self.data['unique_edges'][idx_boundary_odd_edges][:, 1])
        # can be put together with boundary odd vertices to shorten the code
        edge_verts[idx_non_manifold_edges] = self.midpoint_multiple(
            self.data['unique_edges'][idx_non_manifold_edges][:, 0],
            self.data['unique_edges'][idx_non_manifold_edges][:, 1])

        if len(idx_edges) > 0:

            # array of the face indices repeated for every index in idx_zero_crease
            arr_face_indices_zero_crease = np.tile(np.arange(len(self.data['faces'])), len(idx_zero_crease))

            # boolean array where connected tris with zero crease are True; taken from faces_edge_matrix
            connected_tris_zero_crease = (self.data['faces_edges_matrix'][:, idx_zero_crease] == 1).T.flatten()

            # masking the arrays to get only the connected triangular indices
            connected_tris_zero_crease_masked = arr_face_indices_zero_crease[connected_tris_zero_crease].reshape(
                -1, 2)
            # finding vertices opposite to the new edge point
            opposite_verts_zero_crease = np.zeros((len(connected_tris_zero_crease_masked), 2), dtype=int)

            for i in range(len(connected_tris_zero_crease_masked)):
                opposite_verts_zero_crease[i] = np.nonzero(np.sum(self.data['vertex_faces_matrix']
                                                                  [:, connected_tris_zero_crease_masked[i]],
                                                                  axis=1) == 1)[0]

            # calculating the vertex position for the edge points with zero crease
            edge_verts[idx_zero_crease] = self.midpoint_multiple(self.data['unique_edges'][idx_zero_crease][:, 0],
                                                                 self.data['unique_edges'][idx_zero_crease][:, 1],
                                                                 opposite_verts_zero_crease[:, 0],
                                                                 opposite_verts_zero_crease[:, 1])

            # array of the face indices repeated for every index in idx_crease
            array_face_indices_crease = np.tile(np.arange(len(self.data['faces'])), len(idx_crease))
            # boolean array where connected tris with crease are True; taken from faces_edge_matrix
            connected_tris = (self.data['faces_edges_matrix'][:, idx_crease] == 1).T.flatten()
            # masking the arrays to get only the connected tris indices
            connected_tris_masked = array_face_indices_crease[connected_tris].reshape(-1, 2)
            # finding vertices opposite to the new edge point
            opposite_verts = np.zeros((len(connected_tris_masked), 2), dtype=int)
            for i in range(len(connected_tris_masked)):
                sum_around = np.nonzero(np.sum(self.data['vertex_faces_matrix']
                                               [:, connected_tris_masked[i]], axis=1) == 1)[0]
                # if sum_around.shape[0] == 2:
                opposite_verts[i] = sum_around

            # calculating the smooth part of vertex position for the edge points with crease
            edge_points_smooth = self.midpoint_multiple(self.data['unique_edges'][idx_crease][:, 0],
                                                        self.data['unique_edges'][idx_crease][:, 1],
                                                        opposite_verts[:, 0], opposite_verts[:, 1])

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

        return edge_verts, idx_boundary_odd_edges, idx_non_manifold_edges, idx_non_manifold_verts, \
               idx_infinite_crease, idx_crease

    def main(self):
        old_vertex_count = int(len(self.data['vertices']))

        #############################################################################################################
        # Part 1: Generating face points
        # -----------------------------------------------------------
        if self.data['mesh_type'] == 'quadrilateral':
            quad_ctr_list = self.generate_face_points()
        else:
            # triangular meshes don't require face points
            pass

        # Part 2: Generating edge points/ edge children
        # -----------------------------------------------------------

        edge_child_pos, idx_boundary_odd_edges, idx_non_manifold_edges, idx_non_manifold_verts, \
        idx_infinite_crease_edges, idx_crease_edges = self.generate_edge_points()

        idx_boundary_verts = np.unique(self.data['unique_edges'][idx_boundary_odd_edges])
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
        vertex_edge_arr = self.data['vert_edge_list'].toarray()
        for i in range(old_vertex_count):
            connected_edge_mids = np.nonzero(vertex_edge_arr[i] == 1)[0]

            connected_verts = np.nonzero(self.data['vertices_adjacency_matrix'][i] == 1)[0]

            sharp_edges_around_vertex = np.count_nonzero(edges_with_crease[connected_edge_mids][:, 2] == 1,
                                                         keepdims=True)

            connected_quads_faces = (
                list(np.nonzero(np.array(self.data['faces']) == [i])[
                         0]))  # vertex connected to which face (face indices)

            edge_sharpness_idx = edges_with_crease[connected_edge_mids][:, 2]

            edges_sharpness_around_vertex = edge_sharpness_idx
            number_creases_around_vertex = np.count_nonzero((edges_sharpness_around_vertex > 0) &
                                                            (edges_sharpness_around_vertex < 1))

            sum_sharp_around_vertex[i] = np.sum(edges_sharpness_around_vertex)
            number_faces = len(connected_quads_faces)
            number_edges = len(connected_verts)

            if i in idx_non_manifold_verts:
                if number_faces <= 3 or i in idx_boundary_verts:
                    verts_child.append(m1[i])
                else:
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

            elif i in idx_boundary_verts:
                # parent vertices are boundary vertices
                w1 = 0.75
                connected_crease_edges = connected_edge_mids[
                    np.isin(connected_edge_mids, idx_boundary_odd_edges)]
                connected_crease_verts = np.unique(self.data['unique_edges'][connected_crease_edges])
                w3 = np.where(np.isin(connected_verts, connected_crease_verts), 0.125, 0)
                m3 = self.midpoint_edges(connected_verts, w3)
                verts_child.append(m1[i] * w1 + m3)

            # parent and boundary vertices are smooth when no incident edge is sharp/crease
            # a vertex with one sharp edge/crease is a dart and calculated with the smooth vertex rule
            elif sum_sharp_around_vertex[i] == 0 or (number_creases_around_vertex + sharp_edges_around_vertex == 1):
                # parent and child vertices are smooth
                w1_smooth, w3_smooth = vertex_masks.calculate_smooth_mask_for_vertex(number_faces, number_edges)
                m3 = self.midpoint_edges(connected_verts, w3_smooth)
                verts_child.append(m1[i] * w1_smooth + m3)

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

                m3 = self.midpoint_edges(connected_verts, w3_combined)

                verts_child.append(m1[i] * w1_combined + m3)
        self.data['vertices'][:old_vertex_count] = verts_child

        ###############################################################################################################
        # Part 4: Create subdivided model
        # -----------------------------------------------------------

        # 4 child faces per parent face with 3 or for vertices position. Dependent on mesh type.
        new_faces = np.zeros((4 * len(self.data['faces']), len(self.data['faces'][0])), dtype=int)

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

        midpoint_edges = []
        for edge in self.data['edges']:
            midpoint_edges.append(
                edges_directed[np.nonzero((edges_directed[:, 0] == edge[0]) & (edges_directed[:, 1] == edge[1]))][0][2])

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
        crease_around_vertex = np.zeros((len(new_edges), 1))
        mid_edge_crease = np.zeros((len(new_edges), 1))

        idx_vert_with_crease = np.nonzero(sum_sharp_around_vertex > 0)[0]

        for idx in idx_vert_with_crease:
            arr_idx = np.nonzero(new_edges[:, 0] == idx)
            crease_around_vertex[arr_idx] = sum_sharp_around_vertex[idx]

        edges_mid_crease = np.hstack((edges_mid_direction_1, self.data['creases']))

        mid_point_crease = edges_mid_crease[edges_mid_crease[:, 3] > 0][:, 2:]

        new_edges_masked = []

        for i in range(len(mid_point_crease)):
            new_edges_masked.append(
                np.nonzero((new_edges[:, 0] < old_vertex_count) & (new_edges[:, 1] == mid_point_crease[i, 0])))
            mid_edge_crease[new_edges_masked[i]] = mid_point_crease[i, 1]

        # can yield crease values > 1 must be looked into their might be a error in calculation
        new_edges_crease = np.where((mid_edge_crease == 1) | (crease_around_vertex - mid_edge_crease == 1), 1,
                                    np.where(crease_around_vertex * 0.25 + mid_edge_crease * 0.5 - 1 > 0,
                                             crease_around_vertex * 0.25 + mid_edge_crease * 0.5 - 1, 0))
        # correct crease values > 1 by setting them to 1
        new_edges_crease = np.where(new_edges_crease > 1, 1, new_edges_crease)

        # set dynamic vertices for fitting purposes with signed distance function

        if self.data['fitting_method'] is not None:

            dynamic_faces_vertices_edges = property_inhertiage.pass_down_dynamic_faces_vertices(
                self.data['dynamic_faces'], self.data['dynamic_vertices'], self.data['unique_edges'],
                self.data['faces_edges_matrix'], self.data['fitting_method'])

            self.data['dynamic_faces'] = dynamic_faces_vertices_edges[0]
            self.data['dynamic_vertices'] = dynamic_faces_vertices_edges[1]
            self.data['dynamic_edges'] = dynamic_faces_vertices_edges[2]


        # inherit volumes for subdivided faces:
        if self._inherit_volumes:
            volumes_inherited = property_inhertiage.pass_down_volumes(self.data['volumes'])
            self.data['volumes'] = volumes_inherited

        self.data['faces'] = new_faces
        edges = calculation.faces_to_edges(self.data['faces'], self.data['mesh_type'])
        self.data['edges'] = edges
        self.data['unique_edges'] = new_edges
        self.data['creases'] = new_edges_crease

        self.data['faces_edges_matrix'] = data_structure.face_edges_incident(self.data['unique_edges'], edges,
                                                                             self.data['faces'],
                                                                             self.data['mesh_type'])
        graph = data_structure.mesh_graph(self.data['vertices'], self.data['unique_edges'])
        self.data['vert_edge_list'] = data_structure.vert_edge_incident(graph, self.data['unique_edges'],
                                                                        self.data['vertices'])
        self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(graph, self.data['vertices'],
                                                                                  matrix_form=True)
        self.data['vertex_faces_matrix'] = data_structure.vertex_face_incident(self.data['faces'],
                                                                               self.data['vertices'])

        self.data['subdiv_iterations'] += 1  # increase the number of iterations in data array