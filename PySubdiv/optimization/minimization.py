import numpy as np
from PySubdiv.optimization.optimization_backend import init_csv
from PySubdiv.optimization.optimization_backend import objective_functions
from PySubdiv.backend import optimization
from PySubdiv.backend import automatization
from scipy import optimize
from scipy import linalg
import pyswarms as ps
import time


# Variational approach for fitting subdivision surfaces after Wu et. al. [http://dx.doi.org/10.1007/s41095-017-0088-2]


class SurfaceFit(object):

    def __init__(self, mesh, original_mesh, meshes_to_fit=None, use_mesh_data=False, iterations_subdivision=1,
                 variable_edges=True, simple_error=True, a_z=25, epsilon=90):
        self.control_mesh = mesh
        self.original_mesh = original_mesh
        self._use_dynamic_faces = use_mesh_data
        self.meshes_to_fit = meshes_to_fit
        self.iterations_subdivision = iterations_subdivision
        self.variable_edges = variable_edges
        self.simple_error = simple_error
        self.epsilon = epsilon

        # if variable_vertices == 'automatic':
        #     if 'dynamic_vertices' in mesh.data:
        #         self.variable_vertices = variable_vertices
        #     else:
        #         self.variable_vertices = range(len(mesh.vertices))

        # elif isinstance(variable_vertices, list):
        #     self.variable_vertices = variable_vertices
        # else:
        #     self.variable_vertices = range(len(mesh.vertices))

        self.variable_vertices_idx = None
        self.initial_csv = self.initialize_crease_values()
        # self.use_bounds_p = use_bounds_p
        # self.bounds_p = None
        self.v = self.initialize_v(simple_error)
        # self.p = self.initialize_p()
        self.p = None
        self.a_z = a_z
        self.lambda_e = 1
        self.z = np.zeros(self.v.shape)
        self.lambda_z = np.zeros(self.v.shape)
        self.subdivided_mesh = None

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
        use_mesh_data: boolean
            when True, uses dynamic faces defined during creation of the control cage to define sharp edges of the
            control cage.
            When False will deactivate dynamic faces and specific fitting to the defined meshes.
        iterations_subdivision : int
            iterations of subdivision algorithm used during the optimization
        variable_edges : str or list
            indices of edges which crease sharpness value is changed during optimization.
            When True is passed derives variable edges from crease sharpness values. Zero sharpness edges
            are set to be variable.
            When list is passed, will use indices in the list for variable edges.
        a_z : float
            positive number. Step-size of the algorithm.

    """

    @property
    def control_mesh(self):
        return self._control_mesh

    @control_mesh.setter
    def control_mesh(self, mesh):
        self._control_mesh = mesh

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
            if np.all(self._control_mesh.data['dynamic_faces'] == 's'):
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

            dynamic_faces_mask = self._control_mesh.data['dynamic_faces'] != 's'
            dynamic_faces_without_static = self._control_mesh.data['dynamic_faces'][dynamic_faces_mask]
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
            csv_boundaries = init_csv.sharp_creases_from_boundaries(self._control_mesh, self.original_mesh,
                                                                    self._meshes_to_fit)

            csv_angles = init_csv.sharp_creases_from_angles(self._control_mesh, self.epsilon)
            self.initial_csv = csv_boundaries + csv_angles

            if self.variable_edges:
                variable_edges = []
                variable_edges_from_csv = np.nonzero(self.initial_csv == 0)[0]
                variable_edges_from_dynamic_faces = []
                for edge_index, edge in enumerate(self._control_mesh.edges):
                    edge_boolean_mask = []
                    for vertex in edge:
                        if self._control_mesh.data['dynamic_vertices'][vertex] == 's':
                            edge_boolean_mask.append(False)
                        else:
                            edge_boolean_mask.append(True)
                    if all(edge_boolean_mask):
                        variable_edges_from_dynamic_faces.append(edge_index)
                for edge_index in variable_edges_from_csv:
                    if edge_index in variable_edges_from_dynamic_faces:
                        variable_edges.append(edge_index)

                if 'vertex_edges_dictionary' in self._control_mesh.data:
                    pass
                else:
                    self._control_mesh.vertex_edges_dictionary()

                if 'boundary_vertices' in self._control_mesh.data:
                    idx_to_delete = []
                    boundary_vertices_idx = np.nonzero(self._control_mesh.data['boundary_vertices'] == 1)[0]
                    for vert_idx in boundary_vertices_idx:
                        connected_edges = self._control_mesh.data['vertex_edges_dictionary'][vert_idx]
                        for connected_edge in connected_edges:
                            if connected_edge in variable_edges:
                                idx_to_delete.append(connected_edge)
                    idx_to_delete = np.unique(idx_to_delete)
                    self.variable_edges = np.array(variable_edges)
                    indices = np.argwhere(np.isin(self.variable_edges, idx_to_delete))
                    self.variable_edges = np.delete(self.variable_edges, indices)
                else:
                    self.variable_edges = np.array(variable_edges)

            else:
                self.variable_edges = np.array(range(len(self._control_mesh.edges)))

        else:
            self.initial_csv = init_csv.sharp_creases_from_angles(self._control_mesh)

            if self.variable_edges:
                self.variable_edges = np.array(range(len(self.initial_csv)))
            else:
                self.variable_edges = np.array(range(len(self._control_mesh.edges)))

        self._control_mesh.set_crease(self.initial_csv)
        return self.initial_csv

    def initialize_v(self, simple_error):
        if self._use_dynamic_faces:
            if simple_error:
                v = optimization.sdf_with_meshes(self._original_mesh,
                                                 self._control_mesh,
                                                 return_vertices=True)
            else:
                v = optimization.sdf_with_meshes(self._original_mesh,
                                                 self._control_mesh.subdivide(self.iterations_subdivision),
                                                 return_vertices=True)

        else:
            if simple_error:
                v = optimization.sdf(self._original_mesh, self._control_mesh)
            else:
                v = optimization.sdf(self._original_mesh,
                                     self._control_mesh.subdivide(self.iterations_subdivision))
        return v

    # def initialize_p(self):
    #     if self.variable_vertices == 'automatic':
    #         variable_vertices_index = []
    #
    #         if self._use_dynamic_faces:
    #             surface_to_fit = self._meshes_to_fit
    #         else:
    #             surface_to_fit = [0]
    #
    #         vertex_edge_incidence = []
    #         faces_edges_incidence = []
    #         for mesh in self.original_mesh:
    #             vertex_edge_incidence.append(mesh.vertex_edges_dictionary())
    #             faces_edges_incidence.append(mesh.edges_faces_connected())
    #
    #         for i in range(len(self._control_cage.vertices)):
    #             if self.control_mesh.data['dynamic_vertices'][i] == 's':
    #                 pass
    #             elif int(self._control_cage.data['dynamic_vertices'][i]) in surface_to_fit:
    #                 variable_vertices_index.append(i)
    #
    #         if self.use_bounds_p:
    #             self.bounds_p = []
    #             boundaries_control_cage = automatization.find_bounds(self._control_cage)
    #             for variable_vertex in variable_vertices_index:
    #                 counter = 0
    #                 for coord in self._control_cage.vertices[variable_vertex]:
    #                     self.bounds_p.append((boundaries_control_cage[counter], boundaries_control_cage[counter + 1]))
    #                     counter += 2
    #         self.p = self.control_mesh.vertices[np.array(variable_vertices_index)]
    #         self.variable_vertices_idx = variable_vertices_index
    #     else:
    #         self.p = self.control_mesh.vertices[np.array(self.variable_vertices)]
    #         self.variable_vertices_idx = self.variable_vertices
    #     return self.p

    # def optimize(self, number_iteration=5, epsilon_0=1e-5, iterations_swarm=500, nr_particles=10, c1=0.5, c2=0.3,
    #              w=0.9, skip_csv=False):
    def optimize(self, number_iteration=5, epsilon_0=1e-5, skip_csv=False, max_iter=10, max_fev=10):
        start_time = time.time()
        epsilon = np.inf
        iteration = 0
        # self._control_cage.set_crease(np.zeros(len(self._control_cage.edges)))
        if self.iterations_subdivision > 1:
            A_multilevel = []
            for i in range(self.iterations_subdivision):
                A_multilevel.append(self.control_mesh.subdivide(i + 1).data['subdivision_weight_matrix'].toarray())
            for i, matrix in enumerate(reversed(A_multilevel)):
                if i == 0:
                    A = matrix
                else:
                    A = A @ matrix
        else:
            A = self.control_mesh.subdivide().data['subdivision_weight_matrix'].toarray()

        if self.simple_error:
            A = A[:len(self._control_mesh.vertices)]

        if len(self.variable_edges) == 0:
            print('It seems that all crease sharpness values of the edges are constrained, only the control cage can be'
                  ' optimized')
        else:
            init_crease = self._control_mesh.creases[self.variable_edges]

        while epsilon > epsilon_0 and iteration < number_iteration:
            iteration_time = time.time()

            p_0 = self.control_mesh.vertices

            if len(self.variable_edges) == 0 or skip_csv is True:
                new_p = linalg.lstsq(A, self.v)[0]
                self.p = new_p
                self._control_mesh.vertices = new_p

            else:
                new_p = linalg.inv(A.T @ A) @ (A.T @ self.v - A.T @ self.z - (A.T @ self.lambda_z) / self.a_z)
                self.p = new_p
                self._control_mesh.vertices = new_p

                # if "vertex_edges_dictionary" not in self._control_cage.data:
                #     self._control_cage.vertex_edges_dictionary()
                #
                # for v in range(len(self._control_cage.vertices)):
                #     connected_edges = self._control_cage.data["vertex_edges_dictionary"][v]
                #     self.variable_edges = np.array(connected_edges)

                x_max = np.ones(len(self.variable_edges))
                x_min = np.zeros(len(self.variable_edges))

                bounds = (x_min, x_max)
                bounds = []
                for i in range(len(self.variable_edges)):
                    bounds.append((0, 1))

                # options = {'c1': c1, 'c2': c2, 'w': w}

                # initial_pos = np.random.rand(nr_particles, len(init_crease)).round(1)
                # initial_pos[0] = init_crease.flatten()
                # initial_pos[1] = np.zeros(len(init_crease))
                # # initial_pos[2] = np.ones(len(init_crease))
                # crease_idx = np.nonzero(init_crease == 1)[0]
                # initial_pos[3:, crease_idx] = 1
                # print(init_crease)

                args = (self._control_mesh, self.v, self.z, self.lambda_z, self.a_z, self.variable_edges,
                        self.iterations_subdivision, self.simple_error)
                options = {'maxiter': max_iter,
                           'maxfev': max_fev,
                           'disp': True}


                minimized_creases = optimize.minimize(objective_functions.objective_h, init_crease,
                                                      args=args, method="Powell", bounds=bounds, options=options).x
                # minimized_creases = optimize.dual_annealing(objective_functions.objective_h, bounds, args=args).x

                minimized_creases = np.array(minimized_creases).reshape((-1, 1))
                # optimizer = ps.single.GlobalBestPSO(n_particles=nr_particles, dimensions=len(self.variable_edges),
                #                                     options=options, bounds=bounds)
                #
                # cost, result_h = optimizer.optimize(objective_functions.objective_h, iterations_swarm, n_processes=None,
                #                                     mesh=self._control_cage, v=self.v,
                #                                     z=self.z, lambda_z=self.lambda_z, a_z=self.a_z,
                #                                     variable_edges=self.variable_edges,
                #                                     iterations=self.iterations_subdivision)
                #
                # minimized_creases = result_h.T
                self._control_mesh.set_crease(minimized_creases, self.variable_edges)

            result_z = objective_functions.objective_z(self._control_mesh, self.v, self.lambda_z, self.a_z,
                                                       self.lambda_e, self.iterations_subdivision,
                                                       simple_error=self.simple_error)
            self.z = result_z

            self.subdivided_mesh = self._control_mesh.subdivide(self.iterations_subdivision)

            if self.iterations_subdivision > 1:
                A_multilevel = []
                for i in range(self.iterations_subdivision):
                    A_multilevel.append(self.control_mesh.subdivide(i + 1).data['subdivision_weight_matrix'].toarray())
                for i, matrix in enumerate(reversed(A_multilevel)):
                    if i == 0:
                        A = matrix
                    else:
                        A = A @ matrix
            else:
                A = self.control_mesh.subdivide().data['subdivision_weight_matrix'].toarray()

            if self.simple_error:
                A = A[:len(self.control_mesh.vertices)]
                mp = self.subdivided_mesh.vertices[:len(self._control_mesh.vertices)]
            else:
                mp = self.subdivided_mesh.vertices

            self.lambda_z += self.a_z * (self.z - (self.v - mp))

            epsilon = np.linalg.norm((self.p - p_0), ord=2)

            # v_error = self.initialize_v(False)
            # mp_error = self.subdivided_mesh.vertices

            dist = np.array(optimization.sdf_with_meshes(self.original_mesh, self.subdivided_mesh)[2])
            dynamic_vertices_idx = np.nonzero(self.subdivided_mesh.data["dynamic_vertices"] != "s")


            # self.subdivided_mesh.data['deviation'] = v_error - mp_error
            self._control_mesh.ctr_points = range(len(self._control_mesh.vertices))

            print(f"Results for iteration {iteration + 1} of {number_iteration}: ")
            print(f"total deviation of control points: {epsilon}")
            # print(f"total deviation of the meshes: "
            #       f"{np.linalg.norm((v_error - mp_error), ord=2)}")
            # print(f"mean deviation of the meshes: {np.mean(v_error - mp_error)}")
            # print(f"maximal deviation of the meshes: {np.max((v_error - mp_error))}")

            print(f"total deviation of the meshes: {np.sum(dist)}")
            print(f"mean deviation of the meshes: {np.mean(dist)}")
            print(f"standard deviation: {np.std(dist)}")
            print(f"maximal deviation of the meshes: {np.max(dist)}")
            print(f" 25% percentile: {np.quantile(dist, 0.25)}")
            print(f" 75% percentile: {np.quantile(dist, 0.75)}")
            print(f" median: {np.median(dist)}")

            print(f"Number of points: {len(self.subdivided_mesh.vertices)}")
            print("----------------------------------------------------------")
            print("Error only for dynamic vertices")
            print(f"total deviation of the meshes: {np.sum(dist[dynamic_vertices_idx])}")
            print(f"mean deviation of the meshes: {np.mean(dist[dynamic_vertices_idx])}")
            print(f"standard deviation: {np.std(dist[dynamic_vertices_idx])}")
            print(f"maximal deviation of the meshes: {np.max((dist[dynamic_vertices_idx]))}")
            print(f"25% percentile: {np.quantile(dist[dynamic_vertices_idx], 0.25)}")
            print(f"75% percentile: {np.quantile(dist[dynamic_vertices_idx], 0.75)}")
            print(f"median: {np.median(dist[dynamic_vertices_idx])}")
            print(f"Number of points: {len(self.subdivided_mesh.vertices[dynamic_vertices_idx])}")
            print("----------------------------------------------------------")
            print(f"Time iteration: {time.time()-iteration_time} seconds")



            iteration += 1
            if len(self.variable_edges) == 0 or skip_csv is True:
                pass
            else:
                init_crease = self.control_mesh.creases[self.variable_edges]

        # print(f"Optimization finished after iteration {iteration} with a total error of "
        #       f"{np.linalg.norm((v_error - mp_error), ord=2)}")
        print(f"Optimization finished after iteration {iteration} with a total error of "
              f"{np.sum(dist)}")
        print(f"Run time: {time.time() - start_time} seconds")
