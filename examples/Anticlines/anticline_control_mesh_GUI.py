import sys
from PySubdiv.create_control_cage import control_cage
# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"
from qtpy import QtWidgets
from PySubdiv.data import files

import pyvista as pv


# load the original mesh exported from gempy and visualize
anticlineOriginal = files.read("meshes/anticline_joined.obj")

# for creating the control cage we will separate the original mesh into two parts. We can create a list and store
# the two objects there.

anticlineOriginalParts = [pv.read("meshes/anticline_1.obj").rotate_x(90).clean(),
                          pv.read("meshes/anticline_2.obj").rotate_x(90).clean()]


# start the GUI and create or edit the meshes

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = control_cage.PySubdivGUI(anticlineOriginalParts)
    window.main()
    sys.exit(app.exec_())





