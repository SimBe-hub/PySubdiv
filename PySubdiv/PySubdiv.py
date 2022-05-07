from PySubdiv.data import files
from PySubdiv.data.data import data_dictionary
from PySubdiv.data import data_structure
from PySubdiv.backend import utils
from PySubdiv.backend import calculation
from PySubdiv.backend import visualize
from PySubdiv.create_control_cage import control_cage
import numpy as np


class Mesh(object):
    def __init__(self, vertices=None, faces=None, ctr_points=None):

        self.data = data_dictionary()
        self.vertices = vertices
        self.faces = faces
        self.ctr_points = ctr_points

        if self.faces is None:
            self.edges = None
            self.creases = None
            self.face_normals = None
            self.face_centroids = None
            self.vertex_normals = None
        else:
            self.edges = self.edges_unique()
            self.edge_length = self.calculate_edge_length()
            self.average_edge_length = np.mean(self.edge_length)
            self.data['average_edge_length'] = self.average_edge_length
            self.creases = self.set_crease()
            self.face_normals = self.define_face_normals()
            self.face_centroids = self.define_face_centroids()
            self.vertex_normals = self.define_vertex_normals()

    """
      Main class of PySubdiv used to create or load triangular and quadrilateral meshes. The class handles manifold
      as well as non-manifold meshes. The class holds all the important attributes and methods to perform 
      loop subdivision on triangular and Catmull-Clark subdivision on quadrilateral meshes with semi-sharp creases.
      PyVista is used to visualize the meshes. To initialize a simple mesh only the vertices and the faces are needed.
      Subdivision with semi-sharp creases is achieved by setting the crease values for the respective edges.
       
      Attributes
      --------------
      vertices : (n, 3) float or None
        vertices of the mesh. Each vertex has a [x, y, z] Position in cartesian space.
      faces : (n, 3) or (n ,4) int
        vertex indexes that make up quadrilateral or triangular faces. 
      data: dict
        dictionary in which the data of the mesh will be stored
      ctr_points : int or None
        positions of the active control points of the mesh used to change the subdivision surfaces.
        Indices of the vertices are passed and the resp. coordinate is returned. If None is passed all vertices are
        handled as control points. Indices are passed to ctr_points_idx.
      ctr_points_idx : int
        indices of the control points.
      edges : (n,2) int
        unique edges of the mesh. Derived from the faces showing list of vertices (2) forming the edges of the mesh.
      creases : float 
        crease values of the edges for subdivision. Crease values are have a range between 0 - 1. Whereby 0 means smooth
        and 1 means sharp edge. Each edge is assigned one crease value.    
 
    """

    @property
    def vertices(self):
        """
        The vertices of the mesh.

        Returns
        ----------
        vertices : (n, 3) float
          Points in cartesian space referenced by self.faces
        """
        return self.data['vertices']

    @vertices.setter
    def vertices(self, values):
        """
        Assign vertex values to the mesh.

        Parameters
        --------------
        values : (n, 3) float
          Points in space
        """
        if values is None:
            self.data['vertices'] = None
        else:
            values = np.asanyarray(values, order='C', dtype=np.float64)
            if utils.is_shape(values, (-1, 3)):
                self.data['vertices'] = values
            else:
                raise ValueError("Vertices are not in the shape (n,3)")

    @property
    def faces(self):
        """
        The faces of the mesh.

        Returns
        ----------
        faces : (n, 3) or (n, 4) int
          Representing triangles or quadrilaterals which reference self.vertices
        """

        return self.data['faces']

    @faces.setter
    def faces(self, values):
        """
        Set the vertex indexes that make up quadrilateral or triangular faces.

        Parameters
        --------------
        values : (n, 4)/(n, 3) int
          Indexes of self.vertices
        """
        if values is None:
            self.data['faces'] = None
        else:
            values = np.asanyarray(values, order='C', dtype=np.int64)
            self.data['mesh_type'] = utils.mesh_type(values)
            self.data['faces'] = values

    def add_vertex_old(self, vertex):
        """
        Add a vertex to the end of the vertex array

        Parameters
        --------------
        vertex : list with the shape (n, 3) float
          Points in space
        """
        # for future: .data array has to be cleared and faces have to be reindexed

        vertex = np.asanyarray(vertex, order='C', dtype=np.float64)
        if utils.is_shape(vertex, (-1, 3)):
            self.data['vertices'] = np.concatenate((self.data['vertices'], [vertex]), axis=0)
            print('vertex', vertex, 'added at index: ', len(self.data['vertices']) - 1)
        else:
            raise ValueError("Vertices are not in the shape (n,3)")

    def add_vertex(self, face_index, position=None):
        """
        Add a vertex at the passed position or centroid of the passed faces (index). Deletes the passed face and create three new
        faces to fill the hole in the mesh.

        Parameters
        --------------
        face_index : int
            index of face in self.faces
        position: [x, y, z] float or None
            when position is new vertex will be at that position otherwise at the centroid of the face

        """
        # check if passed face index is in the range of the meshes' faces
        if face_index > len(self.faces) - 1:
            print(f"face index {face_index} out of Range for mesh with {len(self.faces)} faces")
        else:
            if position is None:
                position = self.face_centroids[face_index]  # get centroid position of face
            self.vertices = np.append(self.vertices, [position], axis=0)  # add centroid to the vertices
            new_faces = []  # list to store the three new faces
            # create three new faces with the reference from the new vertex and the old vertices building the face
            for counter, idx_vertex in enumerate(self.faces[face_index]):
                new_face = [idx_vertex, len(self.vertices) - 1]
                if counter < 2:
                    new_face.append(self.faces[face_index][counter + 1])
                else:
                    new_face.append(self.faces[face_index][0])
                new_faces.append(new_face)
            # appending the new faces to self.faces
            self.faces = np.append(self.faces, new_faces, axis=0)
            # deleting the old face
            self.faces = np.delete(self.faces, face_index, axis=0)

        self.define_face_centroids()
        self.derive_edges()
        self.edges_unique()

    @property
    def ctr_points(self):
        """
        control points of the mesh

        Returns
        ----------
        vertices : (n, 3) float
          Points in cartesian space referenced by self.faces
        """
        return self.data['ctr_points']

    @ctr_points.setter
    def ctr_points(self, index):
        """
        Set the control points for interactive plotting.

        Parameters
        --------------
        index : int indices of the vertices which should be used as ctr_points
          Points in cartesian space used as control points for interactive plotting
        """
        if self.vertices is None:
            self.data['ctr_points'] = self.vertices
            self.data['ctr_points_idx'] = None

        elif index is None:
            self.data['ctr_points'] = self.vertices
            self.data['ctr_points_idx'] = np.arange(len(self.vertices))
        else:
            self.data['ctr_points'] = self.vertices[index]
            self.data['ctr_points_idx'] = index
        self.ctr_points_idx = self.data['ctr_points_idx']

    def derive_edges(self):
        """
        Edges of the mesh (derived from faces).

        Returns
        ---------
        edges : (n, 2) int
          List of vertex indices making up edges
        """
        self.data['edges'], self.data['edges_face'] = calculation.faces_to_edges(self.data['faces'],
                                                                                 self.data['mesh_type'],
                                                                                 return_index=True)
        return self.data['edges']

    def edges_faces(self):
        """
        Which face does each edge belong to.

        Returns
        ---------
        edges_face : (n, ) int
          Index of self.faces
        """
        # calls function to populate the self._data
        self.derive_edges()
        return self.data['edges_face']

    def edges_sorted(self):
        """
        Sorts the edge list along first axis

        Returns
        ---------
        edges_sorted : (n, 2) int
            List of vertex indices sorted along the first axis
        """
        self.data['sorted_edges'] = np.sort(self.derive_edges())
        return self.data['sorted_edges']

    def edges_unique(self):
        """
        Return the unique edges of the mesh.

        Returns
        ---------
        edges_unique : (n, 2) int
            List of vertex indices making up the unique edges
        """
        edges_sorted = self.edges_sorted()
        self.data['unique_edges'], indices, self.data['inverse_unique_edges'] = np.unique(edges_sorted, axis=0,
                                                                                          return_index=True,
                                                                                          return_inverse=True)
        self.data['index_unique_edges'] = indices
        self.data['unique_edges'] = np.asanyarray([edges_sorted[index] for index in sorted(indices)])
        self.edges = self.data['unique_edges']
        return self.data['unique_edges']

    def set_crease(self, crease_values=None, indices=None):
        """
        Set the crease for the edges which are in the unique_edges array.
        When called the first time creates np.array with zeros of length unique_edges.
        Creases can be set individual when the indices are passed. Multiple creases can be passed as list or tuple.
        Numpy array will raise an error.
        When a list of creases is passed fills the array from top to bottom

        Returns
        --------
            edges_crease: (n,1) float
                np.array with creases for corresponding edge in the unique_edges array
        """

        if 'unique_edges' not in self.data:
            self.edges_unique()

        len_array = len(self.data['unique_edges'])

        if 'creases' not in self.data:
            self.data['creases'] = np.zeros(len_array).reshape(len_array, 1)

        if crease_values is None:
            self.creases = self.data['creases']
            return self.data['creases']
        else:
            if indices is None:
                self.data['creases'][:len(crease_values)] = np.array(crease_values).reshape(-1, 1)
            else:
                if isinstance(indices, (list, np.ndarray)):
                    pass
                else:
                    indices = list(indices)
                self.data['creases'][indices] = np.array(crease_values).reshape(-1, 1)

            self.data['creases'] = np.where(self.data['creases'] > 1, 1, self.data['creases'])
            self.data['creases'] = np.where(self.data['creases'] < 0, 0, self.data['creases'])
            self.creases = self.data['creases']
            return self.data['creases']

    def mesh_graph(self):

        """
        Returns a networkx graph representing the vertices and
        their connections in the mesh.

        Returns
        ---------
        graph : networkx.Graph
            Graph structure vertices and edges
        """
        if 'unique_edges' in self.data:
            pass
        else:
            self.edges_unique()
        self.data['graph'] = data_structure.mesh_graph(self.data['vertices'], self.data['unique_edges'])
        return self.data['graph']

    def vertices_connected(self):
        """
        Returns vertex adjacency matrix (n,n) which shows connected vertices
        If the Graph is already stored in the data will use the stored Graph otherwise mesh_graph is called.
        Returns
        --------
        graph : networkx.Graph.adjacency_matrix
            Adjacency matrix of connected vertices in matrix form
        """
        if "graph" in self.data:
            self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(self.data['graph'],
                                                                                      self.data['vertices'],
                                                                                      matrix_form=True)
        else:

            self.mesh_graph()
            self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(self.data['graph'],
                                                                                      self.data['vertices'],
                                                                                      matrix_form=True)
        return self.data['vertices_adjacency_matrix']

    def vertices_connected_to_list(self):
        """
        Transforms the vertex adjacency matrix (n,n) to a nested list. The index of the list corresponds to the vertex
        the elements in the list to the vertices connection.

        Returns
        --------
        list :
            list of the adjacency matrix of connected vertices
        """
        if 'vertices_adjacency_matrix' in self.data:
            self.data['vertices_adjacency_list'] = data_structure.matrix_to_list(self.data
                                                                                 ['vertices_adjacency_matrix'])
        else:

            self.vertices_connected()
            self.data['vertices_adjacency_list'] = data_structure.matrix_to_list(self.data
                                                                                 ['vertices_adjacency_matrix'])
        return self.data['vertices_adjacency_list']

    def vertices_connected_list(self):
        """
        Returns vertex adjacency list which shows connected vertices.
        If the Graph is already stored in the data will use the stored Graph otherwise mesh_graph is called.

        Returns
        --------
        graph : networkx.Graph.adjacency_matrix
            Adjacency matrix of connected vertices in list form

        """

        if "graph" in self.data:
            self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(self.data['graph'],
                                                                                      self.data['vertices'])
        else:

            self.mesh_graph()
            self.data['vertices_adjacency_matrix'] = data_structure.adjacent_vertices(self.data['graph'],
                                                                                      self.data['vertices'])
        return self.data['vertices_adjacency_matrix']

    def vertices_connected_dictionary(self):
        """
        Returns vertex adjacency in a dictionary, which shows connected vertices
        If the dictionary is already stored in the data will use the stored Graph otherwise mesh_graph is called.
        Returns
        --------
        vertex_connected_dictionary: dict
            Adjacency of connected vertices in dictionary
        """
        if 'vertices_connected_dictionary' not in self.data:
            self.data['vertices_connected_dictionary'] = data_structure.vertex_vertex_dictionary(self.vertices,
                                                                                                 self.edges)
        return self.data['vertices_connected_dictionary']

    def vertices_edges_incidence(self, to_matrix=False):
        """
        Incidence matrix of the mesh, where each row is assigned to a vertex and each column to an edge.
        If the Graph is already stored in the data, will use the stored Graph otherwise mesh_graph is called.

        Returns
        ---------
        graph : Graph.incidence_matrix
            incidence matrix of vertices and edges
        """

        if 'graph' in self.data:
            self.data['vert_edge_list'] = data_structure.vert_edge_incident(self.data['graph'],
                                                                            self.data['unique_edges'],
                                                                            self.data['vertices'])
        else:
            self.mesh_graph()
            self.data['vert_edge_list'] = data_structure.vert_edge_incident(self.data['graph'],
                                                                            self.data['unique_edges'],
                                                                            self.data['vertices'])
        if to_matrix:
            self.data['vert_edge_matrix'] = self.data['vert_edge_list'].toarray()
            return self.data['vert_edge_matrix']
        else:
            return self.data['vert_edge_list']

    def vertices_edges_incidence_to_list(self):
        """
        Transforms the vertices edges incidence matrix to a nested list. The index of the list corresponds to the vertex
        the elements in the list to edges.

        Returns
        --------
        list :
            list of the adjacency matrix of connected vertices
        """
        if 'vert_edge_matrix' in self.data:
            self.data['vertices_edges_incidence_list'] = data_structure.matrix_to_list(self.data
                                                                                       ['vert_edge_matrix'])
        else:

            self.vertices_connected()
            self.vertices_edges_incidence(to_matrix=True)
            self.data['vertices_edges_incidence_list'] = data_structure.matrix_to_list(self.data
                                                                                       ['vert_edge_matrix'])
        return self.data['vertices_edges_incidence_list']

    def vertex_edges_dictionary(self):
        """
        Returns incidence of vertex edges. The key in the dictionary is assigned to a vertex and the values are the indices
        of the connected edges.

        Returns
        ---------
        vertex_edges_dict : dict
            Dictionary of vertices (keys) and connected edges (values)
        """
        if 'vertex_edges_dictionary' not in self.data:
            self.data['vertex_edges_dictionary'] = data_structure.vertex_edges_dict(self.vertices,
                                                                                    self.edges)
        return self.data['vertex_edges_dictionary']

    def faces_edges_incidence(self):
        """
        Returns incidence matrix of faces and edges. Each row is assigned to a face and each column to an edge
        If unique_edges is already stored in the data, will use the stored data otherwise mesh.edges_unique() is called.

        Returns
        ---------
        graph : scipy.coo_matrix
            incidence matrix of faces and edges
        """
        if 'unique_edges' in self.data:
            self.data['faces_edges_matrix'] = data_structure.face_edges_incident(self.data['unique_edges'],
                                                                                 self.data['sorted_edges'],
                                                                                 self.data['faces'],
                                                                                 self.data['mesh_type'])

        else:
            self.edges_unique()
            self.data['faces_edges_matrix'] = data_structure.face_edges_incident(self.data['unique_edges'],
                                                                                 self.data['sorted_edges'],
                                                                                 self.data['faces'],
                                                                                 self.data['mesh_type'])

        return self.data['faces_edges_matrix']

    def faces_edges_incidence_to_list(self, invert=False):
        """
        Transforms the faces edges incidence matrix to a nested list. The index of the list corresponds to the face and
        the elements in the list to the edges surrounding the face.
        If invert = True the index of the list refers to the edge and the elements are the face indices, showing which
        face is adjacent to which other face.

        Returns
        --------
        list :
            Incidence matrix of faces and edges as list
        """
        if 'faces_edges_matrix' in self.data:
            if not invert:
                self.data['faces_edges_incidence_list'] = data_structure.matrix_to_list(self.data
                                                                                        ['faces_edges_matrix'])
                return self.data['faces_edges_incidence_list']
            else:
                self.data['faces_incidence_list'] = data_structure.matrix_to_list(self.data['faces_edges_matrix'],
                                                                                  invert=True)
                return self.data['faces_incidence_list']
        else:
            self.faces_edges_incidence()
            if not invert:
                self.data['faces_edges_incidence_list'] = data_structure.matrix_to_list(self.data
                                                                                        ['faces_edges_matrix'])
                return self.data['faces_edges_incidence_list']
            else:
                self.data['faces_incidence_list'] = data_structure.matrix_to_list(self.data['faces_edges_matrix'],
                                                                                  invert=True)
                return self.data['faces_incidence_list']

    def edges_faces_connected(self):
        """
        Returns incidence of edges and faces in a dictionary. Each key in the dictionary is assigned to an edge and the
        values are the connected faces.
        If unique_edges is already stored in the data, will use the stored data otherwise mesh.edges_unique() is called.

        Returns
        ---------
        edge_faces_dictionary: dict
            incidence of edges and faces
        """
        if 'unique_edges' in self.data:
            self.data['edge_faces_dictionary'] = data_structure.edge_faces_dict(self.data['unique_edges'],
                                                                                self.data['sorted_edges'])

        else:
            self.edges_unique()
            self.data['edge_faces_dictionary'] = data_structure.edge_faces_dict(self.data['unique_edges'],
                                                                                self.data['sorted_edges'])

        return self.data['edge_faces_dictionary']

    def vertices_faces_incidence(self):
        """
        Returns incidence matrix of vertices and faces. Each row is assigned to a vertex and each column to a face

        Returns
        ---------
        graph : scipy.coo_matrix
            incidence matrix of vertices and faces
        """
        self.data['vertex_faces_matrix'] = data_structure.vertex_face_incident(self.data['faces'],
                                                                               self.data['vertices'])
        return self.data['vertex_faces_matrix']

    def vertex_faces_dictionary(self):
        """
        Returns incidence of vertices and faces in a dictionary. Each key is assigned to a vertex and value is the connected
        face index.

        Returns
        ---------
        vertex_face_dictionary : dict
            incidence of vertices and faces
        """
        if 'vertex_faces_dictionary' not in self.data:
            self.data['vertex_faces_dictionary'] = data_structure.vertex_faces_dict(self.data['faces'],
                                                                                    self.data['vertices'])
        return self.data['vertex_faces_dictionary']

    def laplacian_matrix(self):

        """
        Returns the cotangens Laplacian matrix the mesh.
        The Laplacian is the matrix L = D - A, where A is the vertex adjacency matrix and D is the diagonal matrix of
        vertex valence.

        Returns
            laplacian_matrix : |V| x |V| matrix
        """
        self.mesh_graph()
        self.faces_edges_incidence()
        self.vertices_edges_incidence(to_matrix=True)
        if 'vertex_faces_matrix' in self.data:
            self.data['laplacian_matrix'] = data_structure.laplace_matrix(self.vertices,
                                                                          self.faces,
                                                                          self.data['vertex_faces_matrix'],
                                                                          self.data['faces_edges_matrix'],
                                                                          self.data['unique_edges'],
                                                                          self.data['vert_edge_matrix'])
        else:
            self.vertices_faces_incidence()
            self.data['laplacian_matrix'] = data_structure.laplace_matrix(self.vertices,
                                                                          self.faces,
                                                                          self.data['vertex_faces_matrix'],
                                                                          self.data['faces_edges_matrix'],
                                                                          self.data['unique_edges'],
                                                                          self.data['vert_edge_matrix'])
        return self.data['laplacian_matrix']

    def subdivide(self, iteration=1, interactive=False, for_optimization=False):
        """
        Performs subdivision on the mesh with the Catmull-Clark algorithm or loop algorithm depending on the mesh type.

        Parameters
        ----------
        self: mesh
        iteration: int
            How many times the subdivision is performed.
        interactive: bool
            When True the mesh is opened in an interactive pyvista window making it possible to change the position of
            the control points. Calls Mesh.visualize_model_interactive.
        for_optimization: bool
            When True only the position of the vertices are calculated but now connectivity between the vertices are
            calculated, i.e faces, edges, ect. Speeds up the calculation of the algorithm. Should only be used for the
            optimization or when connectivity is not needed. Only 1 level of subdivision possible
        Returns
        ---------
        PySubdiv mesh:
            New PySubdiv mesh with the subdivided vertices and faces.
        """
        if interactive:
            subdiv_surf = self.visualize_mesh_interactive(iteration)
            subdiv_surf.data['vertices'] = np.array(subdiv_surf.data['vertices'], dtype=np.float64)
            subdivided_mesh = Mesh()
            subdivided_mesh.data = subdiv_surf.data

        else:
            if for_optimization:
                subdiv_surf = calculation.call_subdiv_algorithm(self, 1, interactive=False,
                                                                for_optimization=True)
                subdiv_surf.data['vertices'] = np.array(subdiv_surf.data['vertices'], dtype=np.float64)
                subdivided_mesh = Mesh()
                subdivided_mesh.data = subdiv_surf.data
            else:
                subdiv_surf = calculation.call_subdiv_algorithm(self, iteration, interactive=False)
                subdiv_surf.data['vertices'] = np.array(subdiv_surf.data['vertices'], dtype=np.float64)
                subdivided_mesh = Mesh()
                subdivided_mesh.data = subdiv_surf.data

        subdivided_mesh.ctr_points = None
        subdivided_mesh.set_crease()
        subdivided_mesh.edges_unique()
        subdivided_mesh.calculate_edge_length()
        subdivided_mesh.average_edge_length = np.mean(subdivided_mesh.edge_length)
        subdivided_mesh.data['average_edge_length'] = subdivided_mesh.average_edge_length
        subdivided_mesh.define_face_normals()
        subdivided_mesh.define_face_centroids()
        subdivided_mesh.define_vertex_normals()
        return subdivided_mesh

    def simple_subdivision(self, iteration=1):
        """
        Performs simple subdivision on the mesh at the moment only triangular meshes. IMPORTANT: All crease sharpness
        values will be set to zero again.

        Parameters
        ----------
        self: mesh
        iteration: int

        Returns
        ---------
        PySubdiv mesh:
            New PySubdiv mesh with the subdivided vertices and faces.
        """

        subdiv_surf = calculation.call_subdiv_algorithm(self, iteration, interactive=False, simple_subdivision=True)
        subdiv_surf.data['vertices'] = np.array(subdiv_surf.data['vertices'], dtype=np.float64)
        subdivided_mesh = Mesh()
        subdivided_mesh.data = subdiv_surf.data
        subdivided_mesh.ctr_points = None
        subdivided_mesh.edges_unique()
        subdivided_mesh.set_crease(np.zeros(len(subdivided_mesh.edges)))
        subdivided_mesh.calculate_edge_length()
        subdivided_mesh.average_edge_length = np.mean(subdivided_mesh.edge_length)
        subdivided_mesh.data['average_edge_length'] = subdivided_mesh.average_edge_length

        subdivided_mesh.define_face_normals()
        subdivided_mesh.define_face_centroids()
        subdivided_mesh.define_vertex_normals()

        return subdivided_mesh

    @property
    def volumes(self):
        """
        Return volumes of the mesh.

        Parameters
        --------------
        self : PySubdiv mesh object
            mesh on which the volumes are defined

        Returns
        --------------
        volumes: nested list of n volumes and the indices of the faces forming the volume
        """
        volumes = self.data['volumes']
        return volumes

    def define_volumes(self):
        """
        Define volumes for gmesh software. Volumes are set by faces which form a manifold part of the mesh. Faces
        which form a volume can be picked in the viewer by hovering over with the mouse and pressing  the 'P' key.
        After picking the user is asked to provide the index/number of volume which the picked face should be
        assigned to by console input. An integer should be provided.

        The function will return a nested list with n-number of volumes. Whereas the index of the list is the
        index/number of the volume and the elements in the nested list are the indices of the faces forming the
        volume. Important: If a volume index/number one must remember that the returned list is continuous. E.g. if
        volumes 0, 1, 3 are defined the returned list will have the indices 0, 1, 2, so volume 3 becomes index 2 in
        the list.

        Parameters
        --------------
        self : PySubdiv mesh object
            mesh on which the volumes are defined

        """
        if 'faces_incidence_list' not in self.data:
            self.faces_edges_incidence_to_list(invert=True)
        if 'faces_edges_incidence_list' not in self.data:
            self.faces_edges_incidence_to_list()
        volumes = control_cage.define_volumes(self)
        self.data['volumes'] = volumes
        return self.data['volumes']

    def define_face_normals(self):
        """
        Calculate the face normals of the mesh.

        Parameters
        --------------
        self : PySubdiv mesh object
            mesh on which the face normals should be calculated
        Returns
        -------
        face_normals : (n,3)
            face normals for n (number of faces) faces the mesh
        """
        self.model()
        self.data['PyVistaPolyData'] = self.data['PyVistaPolyData'].compute_normals(cell_normals=True,
                                                                                    point_normals=False)
        self.face_normals = self.data['PyVistaPolyData']['Normals']
        self.data['face_normals'] = self.face_normals
        return self.face_normals

    def define_vertex_normals(self):
        """
        Calculate the vertex normals of the mesh by adding up normals of adjacent faces.

        Parameters
        --------------
        self : PySubdiv mesh object
            mesh on which the vertex normals should be calculated
        Returns
        -------
        vertex_normals : (n,3)
            vertex normals for n (number of vertices) vertices the mesh
        """
        self.model()
        vertex_normals = self.data['PyVistaPolyData'].point_normals
        self.vertex_normals = vertex_normals
        self.data['vertex_normals'] = self.vertex_normals
        return self.vertex_normals

    def define_face_centroids(self):
        """
        Calculate the centroid_tris of a triangular face.

        Parameters
        ----------
        self : PySubdiv mesh object
            mesh on which the face centroids should be calculated.

        Returns
        ---------
        centroid_tris : (n,3)
            centroid for each face n in the mesh

        """
        # mesh type check
        if self.data['mesh_type'] == 'triangular':
            centroids = np.zeros(self.faces.shape)
            for face_index, face in enumerate(self.vertices[self.faces]):
                centroids[face_index] = calculation.centroid(face)
            self.face_centroids = centroids
            self.data['face_centroids'] = self.face_centroids
            return self.face_centroids
        else:
            print(f"mesh is from type {self.data['mesh_type']}, only centroids for triangular meshes can be calulated"
                  f"at the moment")

    def recalculate_normals(self):
        """
        Recalculate the vertex and face normals of the mesh.

        Parameters
        --------------
        self : PySubdiv mesh object
            mesh on which the vertex normals should be calculated
        """
        self.define_face_normals()
        self.define_vertex_normals()
        self.define_face_centroids()

    def calculate_edge_length(self):

        length_edges = []
        # vertices indexed by edges
        vertices_indexed_edges = self.vertices[self.edges]
        for edge in vertices_indexed_edges:
            length_edges.append(calculation.distance_vertices(edge[0], edge[1]))
        self.edge_length = length_edges
        self.data['edge_length'] = self.edge_length
        return self.edge_length

    def print_data(self):
        """
        Prints the keys from the data stored in the dictionary to the console.

        Parameters
        ----------
        self: mesh

        Returns
        ---------
        Keys
            Dictionary keys of the stored data.
        """
        keys_data = []
        for key in self.data:
            keys_data.append(key)
        return print(keys_data)

    def model(self):
        """
        Creates PyVista.PolyData object, which can be used for plotting the mesh.

        Parameters
        ----------
        self: mesh

        Returns
        ---------
        PyVista.PolyData
            PyVista object
        """
        model = visualize.create_model(self.data['faces'], self.data['vertices'])
        self.data['PyVistaPolyData'] = model
        return model

    def visualize_mesh(self):
        """
        Creates pyvista.PolyData object and plots the mesh.

        Parameters
        ----------
        self: mesh

        Returns
        ---------
        pyvista.PolyData
            pyvista object
        """
        if 'face_normals' in self.data and 'vertex_normals' in self.data:
            pass
        else:
            self.define_face_normals()
            self.define_face_centroids()
            self.define_vertex_normals()
        model = visualize.print_model(self)
        return model

    def visualize_mesh_interactive(self, iteration=1, additional_meshes=None):
        """
        Performs subdivision on the mesh with the Catmull-Clark or Loop algorithm and opens an interactive
        pyvista window making it possible to change the position of the control points.

        Parameters
        ----------
        self: mesh
        iteration: int
            How many times the subdivision is performed.
        additional_meshes: PySubdiv mesh or list of PySubdiv meshes
            Additional meshes which can be passed to the viewer. Optional
        Returns
        ---------
        Mesh:
            New mesh with the subdivided vertices and faces.

        """
        subdivided_mesh = visualize.visualize_subdivision(self, iteration, additional_meshes)
        return subdivided_mesh

    def save_mesh(self, filename):
        """
        Save a PySubdiv mesh to obj file.
        Parameters
        ----------
        self: PySubdiv mesh object
        filename: (str)
            The string path for the file to save
        """
        files.save(self, filename)

    def save_data(self, filename):
        """
        Save a PySubdiv data dictionary to file

        Parameters
        ----------
        self: PySubdiv mesh object
        filename: (str)
            The string path for the file to save
        """
        files.save_data_dict(self.data, filename)

    def load_data(self, filename):
        """
        Load a PySubdiv data dictionary to mesh

        Parameters
        ----------
        self: PySubdiv mesh object
        filename: (str)
            The string path for the file to load
        """
        dictionary = files.load_dict(filename)
        self.data = dictionary
        self.edges_unique()

        if 'creases' in self.data:
            self.set_crease()
