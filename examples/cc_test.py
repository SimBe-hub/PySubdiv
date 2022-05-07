import sys
from PySubdiv.create_control_cage import  control_cage
# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets


import pyvista as pv



cube = pv.read("meshes/simple_cube.obj")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = control_cage.MyMainWindow(cube)
    window.main()
    sys.exit(app.exec_())