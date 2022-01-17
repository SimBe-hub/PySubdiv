from pysubdiv.main.data import files

import sys

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog, QListWidget, QPushButton, QRadioButton

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vtk_custom_widget


def call_back():
    pass


class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QHBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)

        # enable_point_picker and selection callback

        self.setCentralWidget(self.frame)

        #vlayout.addWidget(self.plainTextEdit)
        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        self.add_mesh_action = QtWidgets.QAction('Import mesh', self)
        self.add_mesh_action.triggered.connect(self.add_mesh)
        fileMenu.addAction(self.add_mesh_action)

        # adding a list widget to store meshes
        self.listwidget = QListWidget()
        self.listwidget.clicked.connect(self.active_mesh)
        vlayout.addWidget(self.listwidget)

        self.mesh_dictionary = {}




        # Button to activate vertex picking
        self.select_vertex_button = QPushButton(self.plotter)
        self.select_vertex_button.setText("Vertex Picking")
        self.select_vertex_button.move(10, 10)
        self.select_vertex_button.clicked.connect(self.vertex_picking)

        self.inactive_mesh = {}
        self.selected_mesh = None
        self.radio_buttons = []

        if show:
            self.show()

    def active_mesh(self, qmodelindex):
        item = self.listwidget.currentItem()
        print(item.text())
        print(self.mesh_dictionary[item.text()])
        self.selected_mesh = self.plotter.renderer.actors[item.text()]
        self.selected_mesh.SetPickable(True)
        print(self.selected_mesh)
        #if item.text() in self.plotter.renderer.actors:
            #self.plotter.remove_actor(item.text())
        #else:
            #self.plotter.add_mesh(self.mesh_dictionary[item.text()].model(), show_edges=True, name=item.text())

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere, show_edges=True)
        self.plotter.reset_camera()

    def add_mesh(self):
        working_path = os.getcwd()
        fname, _ = QFileDialog.getOpenFileName(self, ("Open File"),
                                       f"{working_path}",
                                       (".obj (*.obj)"))
        base_name_from_path = os.path.basename(fname)
        mesh = files.read(fname)
        print(fname)
        self.plotter.add_mesh(mesh.model(), pickable=False, name=base_name_from_path)
        self.plotter.reset_camera()


        self.mesh_dictionary[base_name_from_path] = mesh
        item = QtWidgets.QListWidgetItem(f"{base_name_from_path}")
        self.listwidget.addItem(item)
        widget = QWidget(self.listwidget)
        radio_button = QRadioButton(widget)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(radio_button)
        self.listwidget.setItemWidget(item, widget)
        #radio_button.clicked.connect(lambda ch, text='test': print("\nclicked--> {}".format(text)))
        radio_button.toggled.connect(lambda: self.handleButtonClicked(radio_button, item))
        #print(self.radio_buttons)

    def handleButtonClicked(self, button, item):
        if button.isChecked() == True:
            self.inactive_mesh[item.text()] = self.plotter.renderer.actors[item.text()]
            self.plotter.remove_actor(item.text())
        else:
            print(self.inactive_mesh[item.text()])
            self.plotter.add_actor(self.inactive_mesh[item.text()], name=f"{item.text()}")
            del self.inactive_mesh[item.text()]

        #self.listwidget.insertItem(len(self.mesh_dictionary) - 1, f"{base_name_from_path}")
        #self.listwidget.addItem(QRadioButton)
        # Button to disable mesh



    def vertex_picking(self):
        self.plotter.enable_point_picking(self.position_vertex, use_mesh=True)

    def position_vertex(self, mesh, point):
        vtk_custom_widget.add_position_widget(self.plotter, point, mesh)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())

