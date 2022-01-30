import numpy as np


def is_shape(obj, shape):
    """
     Compare the shape of a numpy.ndarray to a target shape,

     Note that if a list-like object is passed that is not a numpy
     array, this function will not convert it and will return False.

     Parameters
     ------------
     obj :   np.ndarray
       Array to check the shape on
     shape : list or tuple
       Any negative integer will ignored in the comparison

     Returns
     ---------
     shape_ok : bool
       True if shape of obj matches query shape
    """
    if len(obj.shape) == len(shape):  # check if the length of the shape is the same
        # loop through each integer of the two shapes
        for i, target in zip(obj.shape, shape):
            # check if the target shape has negative integers
            if target < 0:
                continue
            # check if the target integers are unequal
            if target != i:
                return False
        # when non of the checks failed obj.shape equals the targeted shape
        return True
    else:
        return False


def mesh_type(faces):
    """
     Returns whether the mesh is triangular or quadrilateral.
     When the mesh is mixed a error will be rised.

     Note that if a list-like object is passed that is not a numpy
     array, this function will not convert it and will return False.

     Parameters
     ------------
     faces :   np.ndarray of faces
       Array to check the shape on

     Returns
     ---------
     shape : string
        Type of the mesh: quadrilateral or triangular
    """

    if is_shape(faces, (-1, 4)):
        return 'quadrilateral'
    elif is_shape(faces, (-1, 3)):
        return 'triangular'
    else:
        raise ValueError("Shapes of faces are mixed. Please use only quadrilateral or triangular faces.")


def check_mesh_type(faces):
    """
     Checks the type of faces in the  PyVista PolyData set. Returns True when all faces are quadrilaterals.

     Parameters
     ------------
     faces:   np.ndarray
        Array of faces in the PyVista PolyData set

     Returns
     ---------
     quadrilaterals  : bool
       True if all faces in the array are quadrilaterals
    """
    idx = 0
    faces_number_vertices = []  # list of number of vertices in each face
    while idx < len(faces):
        faces_number_vertices.append(faces[idx])
        idx = idx + faces[idx] + 1
    if all(np.array(faces_number_vertices) == 4):
        return 'quadrilaterals'
    elif all(np.array(faces_number_vertices) == 3):
        return 'triangles'
    else:
        raise ValueError("Imported mesh features a mix of  quadrilaterals and triangular faces. "
                         "Please use only pure quadrilateral or triangular meshes.")


def is_in_mesh(vertex, target_vertices):
    """
    Checks if the vertex is in the vertices of the mesh.

    Parameters
    ------------
    vertex: [1,3] float
       vertex which should be checked if in mesh.

    target_vertices: [n,3] float
        vertices of the mesh to check for
    Returns
    ---------
    bool  : boolean
      True if in mesh, False if not
    index : int
        index of the vertex in mesh
    """
    hashed_vertex = vertex.flatten.tolist()
    hashed_vertices = []
    for vertex in target_vertices:  # iterate over the vertices of the mesh and calculate hash values
        flatted_vertex = vertex.flatten().tolist()  # flatten and converting to list from numpy array
        hashed_vertices.append(hash(frozenset(flatted_vertex)))  # hashing the list and appending
    if hashed_vertex in hashed_vertices:
        index = np.nonzero(np.array(hashed_vertices) == hashed_vertex)[0]
        return True, index
    else:
        return False, None
