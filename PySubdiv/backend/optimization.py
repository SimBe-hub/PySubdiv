import numpy as np
from scipy.spatial import KDTree
from PySubdiv import PySubdiv
from quad_mesh_simplify import simplify_mesh


def decimate_mesh(mesh, nr_vertices):
    """
    Decimate the mesh using Using Quadric Error Metrics (http://mgarland.org/files/papers/quadrics.pdf)
    --------------
    mesh : PySubdiv mesh

    nr_vertices: int
        Number of vertices after decimation

    Returns
    --------------

    decimated_mesh : PySubdiv mesh
        Decimated PySubdiv mesh

    """
    old_vertices = np.array(mesh.vertices)
    old_faces = np.array(mesh.faces, dtype=np.uint32)
    new_vertices, new_faces = simplify_mesh(old_vertices, old_faces, nr_vertices)
    decimated_mesh = PySubdiv.Mesh(vertices=new_vertices, faces=new_faces)
    return decimated_mesh


def build_tree(mesh):
    """
    Build the KD-Tree from the vertices of a mesh

    Parameters
    --------------
    mesh : PySubdiv mesh object
        mesh from which vertices the KD-Tree is build

    Returns
    --------------
    tree : KD-Tree
    """
    data = mesh.vertices
    tree = KDTree(data)
    return tree


def query_tree(vertices, kd_tree_mesh, k=1):
    """
    Query the tree for the k-th nearest neighbours of the input vertices
    Parameters
    --------------
    vertices : (n,3) float
        vertices to find the k-th nearest neighbours in the kd-tree of the mesh
    kd_tree_mesh : kd-tree of the mesh on which the nearest neighbours should be found

    k : int
        how many nearest neighbours for the vertices should be found

    Returns
    --------------
    idx : int
        index of the nearest neighbour on the mesh
    d : float
        distance to the nearest neighbour
    """
    d, idx = kd_tree_mesh.query(vertices, k=k, p=2)
    return idx, d


def sdf(original_mesh, approximated_mesh, return_distance=False):
    tree = build_tree(original_mesh)
    idx, d = query_tree(approximated_mesh.vertices, tree, k=1)
    vertices = original_mesh.vertices[idx]
    if return_distance:
        return vertices, d
    else:
        return vertices


def sdf_with_dynamic_faces(original_mesh, approximated_mesh):
    poly_original = original_mesh.data['vertices']
    poly_approximated = approximated_mesh.data['vertices']
    tree = KDTree(poly_original)
    vertices = np.zeros(poly_approximated.shape)
    for idx_vertex in range(len(poly_approximated)):
        if approximated_mesh.data['dynamic_vertices'][idx_vertex] == 0:
            vertices[idx_vertex] = poly_approximated[idx_vertex]
        else:
            d, idx = tree.query(poly_approximated[idx_vertex], k=1)
            vertices[idx_vertex] = poly_original[idx]
    return vertices


def sdf_with_meshes(original_mesh, mesh_for_fitting, return_vertices=False):
    if isinstance(original_mesh, list):
        tree = []  # list to hold kd-tree of meshes
        distance = []
        index = []
        index_mesh = []
        vertices = np.zeros(mesh_for_fitting.vertices.shape)
        for mesh in original_mesh:
            tree.append(build_tree(mesh))

        for i in range(len(mesh_for_fitting.vertices)):
            idx_mesh = mesh_for_fitting.data['dynamic_vertices'][i]
            if idx_mesh == 's':
                index_mesh.append(idx_mesh)
                index.append(i)
                distance.append(0)
                if return_vertices:
                    vertices[i] = mesh_for_fitting.vertices[i]
            else:
                idx_mesh = int(idx_mesh)
                idx, d = query_tree(mesh_for_fitting.vertices[i], tree[idx_mesh])
                index.append(idx)
                distance.append(d)
                index_mesh.append(idx_mesh)

                if return_vertices:
                    vertices[i] = original_mesh[idx_mesh].vertices[idx]
        if return_vertices:
            return vertices
        else:
            return index_mesh, index, distance
    else:
        print('original mesh is only one object, please provide list')


