import copy
import numpy as np


def objective_p(p, mesh, v, z, lambda_z, a_z, variable_vertices, iterations=1):
    """
    Objective function to minimize the distance between the control points p on the control cage (provided mesh) and
    vertices on the original mesh (v) regarding to the position of the control points p, crease values are fixed.

    Parameters
    --------------
    p : [n*3] float
        flattened array of control vertices which are compared against v. Provided by the minimizer.
    mesh : PySubdiv mesh object
        control cage on which the subdivision is performed. Vertices position are updated by p.
    v : [n,3] float
        vertices of the original mesh on which distances to p are calculated and minimized.
    z: constrain for the optimization
        z = v-mp, z = 0 in an ideal optimization.
    lambda_z: [n] float
        lagrangian multiplier, scalar.
    a_z: float
        positive number. In the objective function its meaning can be compared to spring constant.
    variable_vertices: [n] int
        indices of vertices which should be changed during minimization.
    iterations : int
        iterations of subdivision performed during optimization.

    Returns
    --------------
    loss : float
        sum of each individual distance v[i] - p[i] which is minimized by the solver
    """
    p = p.reshape(-1, 3)  # reshaping the input array provided by the minimizer to make it usable in PySubdiv mesh
    mesh_calc = copy.deepcopy(mesh)  # copy the control mesh so there are no problems with overwriting
    mesh_calc.data['vertices'][
        np.array(variable_vertices)] = p  # updating the vertices of the control cage with provided positions p
    subdivided_mesh = mesh_calc.subdivide(iterations)  # subdivide the mesh

    mp = subdivided_mesh.data['vertices']  # mp are the vertices of the mesh after subdivision
    diff_v_mp = v - mp  # v-mp
    dot_product_z = []
    # iterate over each scalar and calculate the dot product (first part of objective function)
    for i in range(len(lambda_z)):
        dot_product_z.append(np.dot(lambda_z[i], (z - diff_v_mp)[i]))
    # calculate losses for each vertex pair p and v as distance to the vectors
    losses_z = dot_product_z + a_z / 2 * np.linalg.norm(z - diff_v_mp, axis=1) ** 2
    losses = losses_z
    loss = losses.sum()  # sum up the losses and return the loss which should be minimized
    return loss


def objective_h(h, mesh, v, z, lambda_z, a_z, variable_edges, iterations=1, simple_error=True):
    """
    Objective function to minimize the distance between the control points p on the control cage (provided mesh) and
    vertices on the original mesh (v) regarding the crease values h, control points p are fixed.

    Parameters
    --------------
    h : [n] float range 0.0 - 1.0
        array of crease values which are used during the subdivision Provided by the minimizer.
    mesh : PySubdiv mesh object
        control cage on which the subdivision is performed. Vertices position are updated by p.
    v : [n,3] float
        vertices of the original mesh on which distances to p are calculated and minimized.
    z: constrain for the optimization
        z = v-mp, z = 0 in an ideal optimization.
    lambda_z: [n] float
        lagrangian multiplier, scalar.
    a_z: float
        positive number. In the objective function its meaning can be compared to spring constant.
    variable_edges: [n] int
        indices of edges which should be changed during minimization
    iterations : int
        iterations of subdivision performed during optimization.

    Returns
    --------------
    loss : [n], float
        n one loss for each member of the swarm
        sum of each individual distance v[i] - p[i] which is minimized by the swarm optimizer
    """
    loss = []  # list to store the loss for each member of the swarm
    # for each member of the swarm a subdivision has to be performed. Each with different crease values
    #for i in range(len(h)):

    #h_round = h[i].T  # transpose the provided crease values to fit the needed shape
    h_round = h.T
    mesh_calc = copy.deepcopy(mesh)  # copy the control cage
    mesh_calc.set_crease(h_round, variable_edges)  # set crease values to the copied mesh
    subdivided_mesh = mesh_calc.subdivide(iterations)  # subdivide the mesh one time
    if simple_error:
        mp = subdivided_mesh.data['vertices'][:len(mesh.vertices)]  # mp vertices of the mesh after subdivision position only depends on cv
    else:
        mp = subdivided_mesh.data['vertices']
    diff_v_mp = v - mp  # v-mp: vector between vertices of the original mesh and the control cage
    dot_product = []  # list to store the dot product
    # iterate over each scalar and calculate the dot product (first part of objective function)
    for j in range(len(lambda_z)):
        dot_product.append(np.dot(lambda_z[j], (z - diff_v_mp)[j]))
    # calculate losses for each vertex pair p and v as distance to the vectors
    losses = dot_product + a_z / 2 * np.linalg.norm(z - diff_v_mp) ** 2
    # sum up the losses and return the loss which should be minimized for each member of the swarm
    #loss.append(losses.sum())

    return losses.sum()


def objective_z(mesh, v, lambda_z, a_z, lambda_e, iterations=1, simple_error=True):
    """
    Objective function to minimize the constrain z = v-mp. Minima is a closed form solution when z becomes maximal.
    Position of control vertices p and crease values h are constant

    Parameters
    --------------
    mesh : PySubdiv mesh object
        control cage on which the subdivision is performed. Vertices position are updated by p.
    v : [n,3] float
        vertices of the original mesh on which distances to p are calculated and minimized.
    lambda_z: [n] float
        lagrangian multiplier, scalar.
    a_z : float
        positive number. In the objective function its meaning can be compared to spring constant.
    lambda_e : float
        positive number, Lagrangian of the Lagrangian functional
    iterations : int
        iterations of subdivision performed during optimization.

    Returns
    --------------
    z : [n], float
        array of the constrained for each pair v[i]-mp[i]

    """
    mesh_calc = copy.deepcopy(mesh)  # copy the control cage
    subdivided_mesh = mesh_calc.subdivide(iterations)  # subdivide the copied mesh
    #mp = subdivided_mesh.data['vertices']  # subdivided vertices
    if simple_error:
        mp = subdivided_mesh.data['vertices'][:len(mesh.vertices)]  # subdivided vertices
    else:
        mp = subdivided_mesh.data['vertices']
    diff_v_mp = v - mp  # vector between v[i] and mp[i]
    res = 1 - lambda_e / a_z * np.linalg.norm(diff_v_mp - lambda_z / a_z, axis=1)  # closed form result of z
    z = np.where(res < 0, 0, res)  # if res is smaller than take zero otherwise res
    z = z[:, np.newaxis] * (diff_v_mp - lambda_z / a_z)  # final result for z[i]
    return z
