import numpy as np
from PySubdiv.backend import calculation
from PySubdiv.data import data_structure, data
from PySubdiv.subdivision_algorithms.subdivision_backend import property_inhertiage


class MeshRefiner(object):

    def __init__(self, mesh):

        self.data = data.data_dictionary()
        self.data['vertices'] = mesh.data["vertices"].tolist()
        self.data['faces'] = mesh.data["faces"]
        self.data['mesh_type'] = mesh.data['mesh_type']
        self.edges_list = []
        self.creases_list = []

        if 'unique_edges' in mesh.data:
            self.data['unique_edges'] = mesh.data["unique_edges"]
        else:
            self.data['unique_edges'] = mesh.edges_unique()

        if 'faces_edges_matrix' in mesh.data:
            self.data['faces_edges_matrix'] = mesh.data['faces_edges_matrix']
        else:
            self.data['faces_edges_matrix'] = mesh.faces_edges_incidence()
        if 'edge_faces_dictionary' in mesh.data:
            self.data['edge_faces_dictionary'] = mesh.data['edge_faces_dictionary']
        else:
            self.data['edge_faces_dictionary'] = mesh.edges_faces_connected()

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

        if 'dynamic_faces' in mesh.data:
            self.data['dynamic_faces'] = mesh.data['dynamic_faces']

        if 'dynamic_vertices' in mesh.data:
            self.data['dynamic_vertices'] = mesh.data['dynamic_vertices']

        if 'ctr_points' in mesh.data:
            self.data['ctr_points'] = mesh.data['ctr_points']

        else:
            self.data['ctr_points'] = mesh.ctr_points(self.data['vertices'])

        if 'intersection_verts' in mesh.data:
            self.data['intersection_verts'] = mesh.data['intersection_verts']

        if 'volumes' in mesh.data:
            self.data['volumes'] = mesh.data['volumes']
            self._inherit_volumes = True
        else:
            self._inherit_volumes = False

        if 'subdiv_iterations' in mesh.data:
            self.data['subdiv_iterations'] = mesh.data['subdiv_iterations']
        else:
            self.data['subdiv_iterations'] = 0

        if mesh.data['interactive']:
            self._interactive = True
            self.data['vertices'][:len(self.data['ctr_points'])] = self.data['ctr_points']

        else:
            self._interactive = False

        if 'fitting_method' in mesh.data:
            self.data['fitting_method'] = mesh.data['fitting_method']
        else:
            self.data['fitting_method'] = None

        self._vertices_edges = []

        self.edges_list.append(self.data['unique_edges'])

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

        # calculate edge vertices as average of the old vertices positions
        edge_verts = self.midpoint_multiple(
            self.data['unique_edges'][:, 0],
            self.data['unique_edges'][:, 1])
        return edge_verts

    def refine(self):

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

        edge_child_pos = self.generate_edge_points()

        self.data['vertices'] = np.concatenate((self.data['vertices'], edge_child_pos), axis=0)

        self._vertices_edges = edge_child_pos

        ################################################################################################################
        # Part 3: Updating the position of the old vertices
        # -----------------------------------------------------------
        # No need to calculate new positions for the old vertices

        ###############################################################################################################
        # Part 4: Create subdivided model
        # -----------------------------------------------------------

        # 4 child faces per parent face with 3 or for vertices position. Dependent on mesh type.
        new_faces = np.zeros((4 * len(self.data['faces']), len(self.data['faces'][0])), dtype=int)

        idx_edge_mids = np.arange(len(self.data['unique_edges']))
        first_idx_edge_points = idx_edge_mids + old_vertex_count

        edges_mid = np.concatenate(
            (self.data['unique_edges'], first_idx_edge_points.reshape(len(first_idx_edge_points), 1)),
            axis=1)

        mpoint1 = []
        mpoint2 = []
        np.fill_diagonal(new_faces, self.data['faces'], wrap=True)
        new_faces = np.roll(new_faces, 1, axis=1)

        for curTris, trisVerts in enumerate(self.data['faces'], 0):
            poly_count = len(trisVerts)

            for quadIdx in range(len(trisVerts)):

                idx0 = trisVerts[(quadIdx - 1) % poly_count]
                idx1 = trisVerts[quadIdx]
                idx2 = trisVerts[(quadIdx + 1) % poly_count]

                if idx0 < idx1:
                    mpoint1.append(edges_mid[(edges_mid[:, 0] == idx0) & (edges_mid[:, 1] == idx1)][:, 2])

                else:
                    mpoint1.append(edges_mid[(edges_mid[:, 0] == idx1) & (edges_mid[:, 1] == idx0)][:, 2])

                if idx1 < idx2:
                    mpoint2.append(edges_mid[(edges_mid[:, 0] == idx1) & (edges_mid[:, 1] == idx2)][:, 2])

                else:
                    mpoint2.append(edges_mid[(edges_mid[:, 0] == idx2) & (edges_mid[:, 1] == idx1)][:, 2])

        edges_new_verts = []
        if 'edges_mid_old' not in self.data:
            self.data['num_orig_edges'] = len(self.data['unique_edges'])
            self.data['edges_mid_old'] = edges_mid
            self.data['verts_subdivided_edge'] = edges_mid


        else:

            for curEdge, Verts in enumerate(self.data['edges_mid_old']):
                verts = Verts.tolist()
                new_1 = Verts[[0, 2]]
                new_2 = Verts[[1, 2]]
                verts.append(edges_mid[(edges_mid[:, 0] == new_1[0]) & (edges_mid[:, 1] == new_1[1])][0][2])
                verts.append(edges_mid[(edges_mid[:, 0] == new_2[0]) & (edges_mid[:, 1] == new_2[1])][0][2])
                edges_new_verts.append(verts)
                self.data['verts_subdivided_edge'] = np.array(edges_new_verts)

        new_faces[3::4] = np.reshape(mpoint2, (len(new_faces[3::4]), 3))
        np.fill_diagonal(new_faces, mpoint1, wrap=True)
        new_faces = np.roll(new_faces, 1, axis=1)
        np.fill_diagonal(new_faces, mpoint2, wrap=True)
        new_faces = np.roll(new_faces, 1, axis=1)
        to_edges = np.sort(calculation.faces_to_edges(new_faces, self.data['mesh_type']))
        new_edges, indices = np.unique(to_edges, return_index=True, axis=0)
        new_edges = np.vstack([to_edges[index] for index in sorted(indices)])

        # set dynamic vertices for fitting purposes with signed distance function
        if self.data['fitting_method'] is not None:
            dynamic_faces_vertices_edges = property_inhertiage.pass_down_dynamic_faces_vertices(
                self.data['dynamic_faces'], self.data['dynamic_vertices'], self.data['unique_edges'],
                self.data['edge_faces_dictionary'], self.data['fitting_method'])

            self.data['dynamic_faces'] = dynamic_faces_vertices_edges[0]
            self.data['dynamic_vertices'] = dynamic_faces_vertices_edges[1]
            self.data['dynamic_edges'] = dynamic_faces_vertices_edges[2]

        # inherit volumes for subdivided faces:
        if self._inherit_volumes:
            volumes_inherited = property_inhertiage.pass_down_volumes(self.data['volumes'])
            self.data['volumes'] = volumes_inherited
        else:
            pass

        ###############################################################################################################
        # Part 6: Construct data array
        # -----------------------------------------------------------
        self.data['vertices'] = self.data['vertices'].tolist()

        self.data['faces'] = new_faces
        edges = calculation.faces_to_edges(self.data['faces'], self.data['mesh_type'])
        self.data['unique_edges'] = new_edges
        self.edges_list.append(new_edges)
        self.data['faces_edges_matrix'] = data_structure.face_edges_incident(self.data['unique_edges'], edges,
                                                                             self.data['faces'],
                                                                             self.data['mesh_type'])

        graph = data_structure.mesh_graph(self.data['vertices'], self.data['unique_edges'])
        self.data['vert_edge_list'] = data_structure.vert_edge_incident(graph, self.data['unique_edges'],
                                                                        self.data['vertices'])
        self.data['edge_faces_dictionary'] = data_structure.edge_faces_dict(self.data['unique_edges'], edges)

        self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(graph, self.data['vertices'],
                                                                                  matrix_form=True)
        self.data['vertex_faces_matrix'] = data_structure.vertex_face_incident(self.data['faces'],
                                                                               self.data['vertices'])
        self.data['edges_mid_old'] = edges_mid

        self.data['subdiv_iterations'] += 1  # increase the number of iterations in data array
