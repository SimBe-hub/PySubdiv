import numpy as np
import init_csv
import objective_functions
import optimization
import automatization
from scipy import optimize
import pyswarms as ps
import main
import pyvista as pv

class mesh_optimizer(object):

    def __init__(self, mesh, original_mesh, meshes_to_fit=None, use_dynamic_faces=False, iterations_subdivision=1,
                 variable_edges='automatic', variable_vertices='automatic', use_bounds_p=False, a_z=1, lambda_e=1):
        self.control_cage = mesh
        self.original_mesh = original_mesh
        self._use_dynamic_faces = use_dynamic_faces
        self.meshes_to_fit = meshes_to_fit
        self.iterations_subdivision = iterations_subdivision
        self.variable_edges = variable_edges

        if variable_vertices == 'automatic':
            if 'dynamic_vertices' in mesh.data:
                self.variable_vertices = variable_vertices
            else:
                self.variable_vertices = range(len(mesh.vertices))
        else:
            self.variable_vertices = variable_vertices

        self.variable_vertices_idx = None
        self.initial_csv = self.initialize_crease_values()
        self.use_bounds_p = use_bounds_p
        self.bounds_p = None
        self.v = self.initialize_v()
        self.p = self.initialize_p()
        self.a_z = a_z
        self.lambda_e = lambda_e
        self.z = np.zeros(self.v.shape)
        self.lambda_z = np.zeros(self.v.shape)
        """
        This class is used to initialize and perform the optimization routine on the control cage and the
        original mesh. First some attributes have to be initialized to 

        Attributes
        --------------
        mesh : PySubdiv mesh object
            control cage which should be optimized to get the best fitted approximated mesh.
        original_mesh: PySubdiv mesh object
            mesh or list of meshes to which the approximated mesh should be fitted to.
        meshes_to_fit: list
            list of indices of the meshes when specific indices are passed will only use passed indices for specific
            fitting, when no or empty list is passed will use all meshes for specific fitting.
        use_dynamic_faces: boolean
            when True, uses dynamic faces defined during creation of the control cage to define sharp edges of the
            control cage.
            When False will deactivate dynamic faces and specific fitting to the defined meshes.
        iterations_subdivision : int
            iterations of subdivision algorithm used during the optimization
        variable_edges : str or list
            indices of edges which crease sharpness value is changed during optimization.
            When 'automatic' is passed derives variable edges from crease sharpness values. Zero sharpness edges
            are set to be variable.
            When list is passed, will use indices in the list for variable edges.
        variable_vertices: str or list
            indices of vertices which position is changed during optimization.
            When 'automatic' is passed derives variable vertices from the control cage's attribute 
            'dynamic_vertices', either from indices of dynamic meshes or dynamic faces 
            When list is passed, will use indices in the list for variable vertices.
        use_bound_p : boolean
            when True will bound the control points to the boundary box of the mesh
        a_z : float
            positive number. In the objective function its meaning can be compared to spring constant.
        lambda_e : float
            positive number, Lagrangian of the Lagrangian functional
    """

    @property
    def control_cage(self):
        return self._control_cage

    @control_cage.setter
    def control_cage(self, mesh):
        self._control_cage = mesh

    @property
    def original_mesh(self):
        return self._original_mesh

    @original_mesh.setter
    def original_mesh(self, mesh):
        self._original_mesh = mesh

    @property
    def meshes_to_fit(self):
        return self._meshes_to_fit

    @meshes_to_fit.setter
    def meshes_to_fit(self, idx_meshes):
        if not self._use_dynamic_faces:
            self._meshes_to_fit = False
        elif self._use_dynamic_faces:
            if np.all(self._control_cage.data['dynamic_faces'] == 's'):
                raise AssertionError('All faces of the mesh are set to static.')
            if isinstance(idx_meshes, (type(None), list)):
                if idx_meshes is None or len(idx_meshes) == 0:
                    if self.get_number_meshes() == -1:
                        self.original_mesh = [self._original_mesh]
                        self._meshes_to_fit = [0]
                    else:
                        self._meshes_to_fit = list(range(len(self.original_mesh)))
                elif max(idx_meshes) <= abs(self.get_number_meshes()) - 1:
                    self._meshes_to_fit = idx_meshes
                else:
                    raise IndexError(f"highest index {max(idx_meshes)} (Parameter meshes_to_fit) is out of range "
                                     f"for the number ({abs(self.get_number_meshes())}) of passed meshes")
            else:
                raise TypeError('indices of meshes should be list or None.')

            dynamic_faces_mask = self._control_cage.data['dynamic_faces'] != 's'
            dynamic_faces_without_static = self._control_cage.data['dynamic_faces'][dynamic_faces_mask]
            if int(max(dynamic_faces_without_static)) > abs(self.get_number_meshes()):
                raise AssertionError(f'The maximal mesh number ({max(dynamic_faces_without_static)}) of the dynamic '
                                     f'faces exceeds number of meshes ({abs(self.get_number_meshes())}) passed to '
                                     f'the optimizer ')

    def get_number_meshes(self):
        if isinstance(self._original_mesh, (list, tuple)):
            return len(self._original_mesh)
        else:
            return -1

    def initialize_crease_values(self):
        if self._use_dynamic_faces:
            csv_boundaries = init_csv.sharp_creases_from_boundaries(self._control_cage, self.original_mesh,
                                                                    self._meshes_to_fit)

            csv_angles = init_csv.sharp_creases_from_angles(self._control_cage)
            self.initial_csv = csv_boundaries + csv_angles

            if self.variable_edges == 'automatic':
                variable_edges = []
                variable_edges_from_csv = np.nonzero(self.initial_csv == 0)[0]
                variable_edges_from_dynamic_faces = []
                for edge_index, edge in enumerate(self._control_cage.edges):
                    edge_boolean_mask = []
                    for vertex in edge:
                        if self._control_cage.data['dynamic_vertices'][vertex] == 's':
                            edge_boolean_mask.append(False)
                        else:
                            edge_boolean_mask.append(True)
                    if all(edge_boolean_mask):
                        variable_edges_from_dynamic_faces.append(edge_index)
                for edge_index in variable_edges_from_csv:
                    if edge_index in variable_edges_from_dynamic_faces:
                        variable_edges.append(edge_index)
                print(variable_edges)
                self.variable_edges = np.array(variable_edges)

        else:
            self.initial_csv = init_csv.sharp_creases_from_angles(self._control_cage)
            self.variable_edges = np.array(range(len(self.initial_csv)))
        self._control_cage.set_crease(self.initial_csv)
        return self.initial_csv

    def initialize_v(self):
        if self._use_dynamic_faces:
            self.v = optimization.sdf_with_meshes(self._original_mesh,
                                                  self._control_cage.subdivide(self.iterations_subdivision),
                                                  return_vertices=True)
        else:
            self.v = optimization.sdf(self._original_mesh, self._control_cage.subdivide(self.iterations_subdivision))
        return self.v

    def initialize_p(self):
        if self.variable_vertices == 'automatic':
            variable_vertices_index = []

            if self._use_dynamic_faces:
                surface_to_fit = self._meshes_to_fit
            else:
                surface_to_fit = [0]

            vertex_edge_incidence = []
            faces_edges_incidence = []
            for mesh in self.original_mesh:
                vertex_edge_incidence.append(mesh.vertices_edges_incidence_to_list())
                faces_edges_incidence.append(mesh.faces_edges_incidence_to_list(invert=True))

            for i in range(len(self._control_cage.vertices)):
                if self.control_cage.data['dynamic_vertices'][i] == 's':
                    pass
                elif int(self._control_cage.data['dynamic_vertices'][i]) in surface_to_fit:
                    variable_vertices_index.append(i)

            if self.use_bounds_p:
                self.bounds_p = []
                boundaries_control_cage = automatization.find_bounds(self._control_cage)
                for variable_vertex in variable_vertices_index:
                    counter = 0
                    for coord in self._control_cage.vertices[variable_vertex]:
                        self.bounds_p.append((boundaries_control_cage[counter], boundaries_control_cage[counter + 1]))
                        counter += 2
            self.p = self.control_cage.vertices[np.array(variable_vertices_index)]
            self.variable_vertices_idx = variable_vertices_index
        else:
            self.p = self.control_cage.vertices[np.array(self.variable_vertices)]
            self.variable_vertices_idx = self.variable_vertices
        return self.p

    def optimize(self, number_iteration=5, epsilon_0=1e-5, iterations_swarm=500, nr_particles=10, c1=0.5, c2=0.3, w=0.9):

        epsilon = np.inf
        iteration = 0
        if len(self.variable_edges) == 0:
            print('It seems that all crease sharpness values of the edges are constrained, only the control cage can be'
                  'optimized')
        else:
            init_crease = self._control_cage.creases[self.variable_edges]
        while epsilon > epsilon_0 and iteration < number_iteration:
            p_0 = self.p
            result_p = optimize.minimize(objective_functions.objective_p, self.p, method='SLSQP',
                                         args=(self._control_cage, self.v, self.z, self.lambda_z, self.a_z,
                                               self.variable_vertices_idx, self.iterations_subdivision),
                                         bounds=self.bounds_p, options={'disp': True})

            self.p = result_p.x.reshape(-1, 3)
            self._control_cage.vertices[self.variable_vertices_idx] = self.p

            if len(self.variable_edges) == 0:
                pass
            else:
                x_max = np.ones(len(self.variable_edges))
                x_min = np.zeros(len(self.variable_edges))

                bounds = (x_min, x_max)

                options = {'c1': c1, 'c2': c2, 'w': w}

                initial_pos = np.random.rand(nr_particles, len(init_crease)).round(1)
                initial_pos[0] = init_crease.flatten()
                initial_pos[1] = np.zeros(len(init_crease))
                initial_pos[2] = np.ones(len(init_crease))
                crease_idx = np.nonzero(init_crease == 1)[0]
                initial_pos[3:, crease_idx] = 1

                optimizer = ps.single.GlobalBestPSO(n_particles=nr_particles, dimensions=len(self.variable_edges),
                                                    options=options, bounds=bounds, init_pos=initial_pos)

                cost, result_h = optimizer.optimize(objective_functions.objective_h, iterations_swarm, n_processes=2,
                                                    mesh=self._control_cage, v=self.v,
                                                    z=self.z, lambda_z=self.lambda_z, a_z=self.a_z,
                                                    variable_edges=self.variable_edges,
                                                    iterations=self.iterations_subdivision)

                minimized_creases = result_h.T
                self._control_cage.set_crease(minimized_creases, self.variable_edges)

            result_z = objective_functions.objective_z(self._control_cage, self.v, self.lambda_z, self.a_z,
                                                       self.lambda_e, self.iterations_subdivision)
            self.z = result_z

            subdivided_mesh = self._control_cage.subdivide(self.iterations_subdivision)

            mp = subdivided_mesh.data['vertices']

            self.lambda_z += self.a_z * (self.z - (self.v - mp))
            self.initialize_v()
            epsilon = np.linalg.norm((self.p - p_0), ord=2)

            print(f"total deviation of control points: {epsilon}")
            print(f"total deviation of the meshes: "
                  f"{np.linalg.norm((self.v - mp), ord=2)}")
            print(f"mean deviation of the meshes: {np.mean(self.v - mp)}")
            print(f"maximal deviation of the meshes: {(self.v - mp)}")
            iteration += 1
            if len(self.variable_edges) == 0:
                pass
            else:
                init_crease = self.control_cage.creases[self.variable_edges]
        print(f"Optimization finished after iteration {iteration} with a total error of "
              f"{np.linalg.norm((self.v - mp), ord=2)}")
