import sys
from PySubdiv.create_control_cage import control_cage
from PySubdiv.data import files
# Setting the Qt bindings for QtPy
import os
import numpy as np
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets


import pyvista as pv



test = files.read("meshes/simple_cube.obj")
test.visualize_mesh()
test.set_crease(np.ones(len(test.edges))*0.6)
#test.set_crease([1],[1])
test = test.subdivide(2)
test.visualize_mesh()
test.save_mesh("meshes/simple_cube_3.obj")
print(test.vertices)

edges_mid_point = []
for edge in test.edges:
    edges_mid_point.append((test.vertices[edge[0]] + test.vertices[edge[1]]) / 2)
edges_mid_point = np.array(edges_mid_point)
print(edges_mid_point)

p = pv.Plotter()
p.add_mesh(test.model(), show_edges=True)

label = np.zeros(len(test.vertices))

p.add_point_labels(edges_mid_point, test.creases)
p.show()

exit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = control_cage.MyMainWindow(cube)
    window.main()
    sys.exit(app.exec_())