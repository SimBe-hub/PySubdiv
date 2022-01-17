import numpy as np
from pysubdiv.optimization.optimization_backend import init_csv
from pysubdiv.optimization.optimization_backend import objective_functions
from pysubdiv.backend import optimization
from pysubdiv.backend import automatization
from scipy import optimize
import pyswarms as ps


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
        meshes_to_fit: tuple
            indices of the meshes to fit the approximated mesh to.
        use_dynamic_faces: boolean
            when True, uses dynamic faces defined during creation of the control cage to define sharp edges of the
            control cage.
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
        if idx_meshes is None:
            self._meshes_to_fit = None
        elif self._use_dynamic_faces:
            print('Dynamic faces are set as True. Mesh ID is ignored until set to False.')
        elif isinstance(idx_meshes, tuple):
            if self.get_number_meshes() != 1:
                self._meshes_to_fit = idx_meshes
            else:
                raise TypeError('original meshes are not list.')

        else:
            raise TypeError('indices of meshes should be tuple.')

    def get_number_meshes(self):
        if isinstance(self._original_mesh, list):
            return len(self._original_mesh)
        else:
            return 1

    def initialize_crease_values(self):
        if self._use_dynamic_faces:
            self.initial_csv = init_csv.sharp_creases_from_dynamic_faces(self._control_cage)
            if self.variable_edges == 'automatic':
                self.variable_edges = np.nonzero(self.initial_csv == 0)[0]
        elif self._meshes_to_fit is None:
            self.initial_csv = init_csv.sharp_creases_from_angles(self._control_cage)
            self.variable_edges = np.array(range(len(self.initial_csv)))
        else:
            self.initial_csv = init_csv.sharp_creases_from_dynamic_meshes(self._control_cage, self._meshes_to_fit)
            if self.variable_edges == 'automatic':
                self.variable_edges = np.nonzero(self.initial_csv == 0)[0]
        self._control_cage.set_crease(self.initial_csv)
        return self.initial_csv

    def initialize_v(self):
        if self._use_dynamic_faces:
            self.v = optimization.sdf_with_dynamic_faces(self._original_mesh,
                                                         self._control_cage.subdivide(self.iterations_subdivision))

        elif self.get_number_meshes() > 1:
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
                surface_to_fit = [1]
            else:
                surface_to_fit = self._meshes_to_fit

            if self.use_bounds_p:
                self.bounds_p = []
                boundaries_control_cage = automatization.find_bounds(self._control_cage)
                for i in range(len(self._control_cage.vertices)):
                    if self._control_cage.data['dynamic_vertices'][i] in surface_to_fit:
                        counter = 0
                        variable_vertices_index.append(i)
                        for coord in self._control_cage.vertices[i]:
                            self.bounds_p.append((boundaries_control_cage[counter],
                                                  boundaries_control_cage[counter + 1]))
                            counter += 2
            else:
                for i in range(len(self._control_cage.vertices)):
                    if self._control_cage.data['dynamic_vertices'][i] in surface_to_fit:
                        variable_vertices_index.append(i)

            self.p = self.control_cage.vertices[np.array(variable_vertices_index)]
            self.variable_vertices_idx = variable_vertices_index
        else:
            self.p = self.control_cage.vertices[np.array(self.variable_vertices)]
            self.variable_vertices_idx = self.variable_vertices
        return self.p

    def optimize(self, number_iteration=5, epsilon_0=1e-5, iterations_swarm=500, nr_particles=10, c1=0.5, c2=0.3, w=0.9):

        epsilon = np.inf
        iteration = 0
        init_crease = self._control_cage.creases[self.variable_edges]
        while epsilon > epsilon_0 and iteration < number_iteration:
            p_0 = self.p

            result_p = optimize.minimize(objective_functions.objective_p, self.p, method='SLSQP',
                                         args=(self._control_cage, self.v, self.z, self.lambda_z, self.a_z,
                                               self.variable_vertices_idx, self.iterations_subdivision),
                                         bounds=self.bounds_p)

            self.p = result_p.x.reshape(-1, 3)
            self._control_cage.vertices[self.variable_vertices_idx] = self.p

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
            iteration += 1
            init_crease = self.control_cage.creases[self.variable_edges]


        print('finished')
