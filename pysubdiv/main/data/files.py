import pyvista as pv
import numpy as np
from pysubdiv.main import main
from pysubdiv.backend import utils
from pysubdiv.main.data import data_structure
import pickle


def read(filename):
    """
    Load a file and transforms it to a PySubdiv mesh.
    Parameters
    ----------
    filename: (str)
        The string path from the file to read
    Returns
    ---------
    PySubdiv mesh
    """

    if filename.endswith('.obj'):
        pass
    else:
        filename += ".obj"
    file = pv.read(filename)  # import file

    mesh_type = utils.check_mesh_type(file.faces)  # check if all faces are quadrilateral
    try:
        file_merged = file.merge(file).extract_geometry()
    except ValueError:
        file_merged = file.extract_geometry()
    if mesh_type == 'quadrilaterals':
        vertices = file_merged.points  # extract vertices array
        faces = file_merged.faces.reshape(-1, 5)[:, 1:]  # reshaping the PyVista.PolyData faces array
        return main.Mesh(vertices, faces[:int(len(faces) / 2)])  # returning the PySubdiv mesh object
    elif mesh_type == 'triangles':
        vertices = file_merged.points  # extract vertices array
        faces = file_merged.faces.reshape(-1, 4)[:, 1:]  # reshaping the PyVista.PolyData faces array
        return main.Mesh(vertices, faces[:int(len(faces) / 2)])  # returning the PySubdiv mesh object

    else:
        raise ValueError("Imported mesh features none quadrilaterals faces. "
                         "Please use only fully quadrilaterals meshes.")


def save(mesh, filename):
    """
    Save a PySubdiv mesh to vtk file.
    Parameters
    ----------
    mesh: PySubdiv mesh object
    filename: (str)
        The string path for the file to save
    """
    if filename.endswith('.obj'):
        pass
    else:
        filename += ".obj"
    file = mesh.model()  # create PyVista PolyData object from the PySubdiv mesh
    pv.save_meshio(filename, file)  # save the PyVista PolyData


def save_data_dict(dictionary, filename):
    """
    Save a dictionary to python's pickle format.
    Parameters
    ----------
    dictionary: dictionary

    filename: (str)
        The string path for the file to save
    """
    if filename.endswith('.pkl'):
        with open(f"{filename}", 'wb') as file:
            pickle.dump(dictionary, file)
    else:
        with open(f"{filename}.pkl", 'wb') as file:
            pickle.dump(dictionary, file)


def load_dict(filename):
    """
    Save a dictionary to python's pickle format.
    Parameters
    ----------
    filename: (str)
        The string path for the file to load
    Returns
    ---------
    dictionary : saved dictionary
    """
    if filename.lower().endswith('.pkl'):
        with open(f"{filename}", 'rb') as file:
            dictionary = pickle.load(file)
    else:
        with open(f"{filename}.pkl", 'rb') as file:
            dictionary = pickle.load(file)
    return dictionary
