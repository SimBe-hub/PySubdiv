from PySubdiv.data import files
import numpy as np

# load a mesh
faultDomainControlCage = files.read('../FaultDomain/controlCage/FaultDomainControlCage.obj')
# load data of the mesh
faultDomainControlCage.load_data('../FaultDomain/controlCage/FaultDomainControlCageData.pkl')
faultDomainControlCage.visualize_mesh()

# print array of dynamic faces, the index of the array corresponds to the face and the element is
# the index of the mesh the face should be fitted to 's' means static no fit
print("Dynamic faces: \n", faultDomainControlCage.data['dynamic_faces'])
# print array of dynamic vertices, the index of the array corresponds to the vertex and the element is
# the index of the mesh the vertex should be fitted to 's' means static -> no fit

print("Dynamic vertices: \n", faultDomainControlCage.data['dynamic_vertices'])

# find indices of vertices to be fitted to mesh_3 with numpy.nonzero -> return indices where condition is true
# Caution: mesh indices are stored as strings
indices_vertices_mesh_3 = np.nonzero(faultDomainControlCage.data['dynamic_vertices'] == '3')
print("Index of dynamic vertices fitted to mesh_3: \n", indices_vertices_mesh_3)
# When doing subdivision these properties are inherited to the subdivided mesh
faultDomainControlCage_simple_subdivided = faultDomainControlCage.simple_subdivision(2)
print("Dynamic vertices: \n", faultDomainControlCage_simple_subdivided.data['dynamic_vertices'])
