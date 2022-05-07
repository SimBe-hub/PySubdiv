import os
import pyvista.core.composite
from PySubdiv.backend import automatization
import pyvista as pv
import numpy as np
from PySubdiv.backend import calculation
from PySubdiv.backend import optimization
from PySubdiv import PySubdiv
from PySubdiv.create_control_cage import vtk_custom_widget
import enum
import copy
import pymeshlab
from PySubdiv.data import files, data_structure
import easygui

from pyvistaqt import QtInteractor, MainWindow

from qtpy import QtWidgets
from qtpy.QtWidgets import QLabel, QPushButton, QInputDialog, QMessageBox, QComboBox, QListWidget
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class States(enum.Enum):
    picking_selection = 1
    deleting_selection = 2
    polygon_cutting = 3
    mesh_decimation = 4
    mesh_subdivision = 5
    constrained_selection = 6


def create_control_cage(mesh, find_vertices=False, calc_intersection=False, add_boundary_box=False,
                        use_dynamic_faces=True, simple_subdivision=False, iteration_subdivision=1,
                        render_style='wireframe', color='green'):
    p = pv.Plotter()  # initialize the PyVista.Plotter
    points_1 = []  # list to store vertices which will used to create the point cloud
    points_2 = []  # list to store vertices which will used to create second point cloud
    added_triangles = set()  # set to hold triangles, to check for duplicates
    invisible_meshes = []
    selected_points = []  # list for storing tris/quads of vertex indices
    line = []  # array to store line mesh for plotting
    labels = []  # array to store vertex index labels for plotting
    points = []  # list to store vertex label points for plotting
    dynamic_face = []  # list to store dynamic faces
    dynamic_vertices_dict = {}  # dictionary to store dynamic vertices
    vertices_fit = []  #
    control_cage_faces = []
    faces = []

    p_cloud_complete = pv.PolyData()
    p_cloud = pv.PolyData()

    # check if one mesh or list of meshes are passed and create Pyvista models
    if isinstance(mesh, list):
        points_for_labels = []
        original_model = []
        for part_mesh in mesh:
            original_model.append(part_mesh.model())
            com = calculation.center_of_mass(part_mesh)
            part_mesh_kdtree = optimization.build_tree(part_mesh)
            points_for_labels.append(part_mesh.vertices[optimization.query_tree(com, part_mesh_kdtree)[0]])
        label_meshes = pv.PolyData(points_for_labels)
        label_meshes['label'] = ['mesh_' + str(i) for i in range(len(mesh))]
        p.add_point_labels(label_meshes, "label", point_size=20, font_size=36, tolerance=0.01,
                           show_points=True, name='label_mesh')
    else:
        original_model = [mesh.model()]
        mesh = [mesh]

    for mesh_part in mesh:
        if not mesh_part.data["mesh_type"] == 'triangular':
            print('At the moment only triangular meshes are supported')
            return

    # If true: Find vertices which are pickable for creating the control cage
    if find_vertices:
        if isinstance(mesh, list):
            boundary_vertices_arr = []
            corner_vertices_arr = []
            for part_mesh in mesh:
                corner_vertices, boundary_vertices = automatization.find_corner_vertices(part_mesh)
                boundary_vertices_arr.append(boundary_vertices)
                corner_vertices_arr.append(corner_vertices)

            boundary_vertices = np.concatenate(boundary_vertices_arr)
            corner_vertices = np.concatenate(corner_vertices_arr)
            if len(boundary_vertices) + len(corner_vertices) == 0:
                print('Is your input mesh watertight?\n'
                      'Boundary and corner vertices can only be found on separated meshes at the moment')
                points_1.append([mesh[0].vertices[0]])
                points_2.append([mesh[0].vertices[0]])
                p_cloud = pv.PolyData(mesh[0].vertices[0])
                p_cloud_complete = pv.PolyData(mesh[0].vertices[0])

            else:
                points_2.append(corner_vertices)
                points_1.extend([corner_vertices, boundary_vertices])

                p_cloud = pv.PolyData(corner_vertices)
                p_cloud_complete = pv.PolyData(corner_vertices) + pv.PolyData(boundary_vertices)

        if calc_intersection:
            if len(mesh) > 1:
                intersection_vertices = []
                compared_meshes = []
                for mesh_1 in mesh:
                    for mesh_2 in mesh:
                        if mesh_1 == mesh_2:
                            continue
                        elif hash(frozenset([mesh_1, mesh_2])) in compared_meshes:
                            continue
                        else:
                            intersection_vertices.append(automatization.find_intersection_points(mesh_1, mesh_2))
                            compared_meshes.append((hash(frozenset([mesh_1, mesh_2]))))
                intersection_vertices = np.concatenate(intersection_vertices)
                p_cloud_complete = pv.PolyData(intersection_vertices + p_cloud_complete)
                p_cloud = pv.PolyData(intersection_vertices + p_cloud)
                points_1.insert(1, intersection_vertices)
                points_2.append(intersection_vertices)

            else:
                print('Input mesh is only one object.\n'
                      'At the moment the intersection can only be found for separated meshes')

        if add_boundary_box:
            boundary_box = automatization.define_boundary_box(mesh)
            points_1.insert(2, boundary_box)
            points_2.append(boundary_box)
            p_cloud = pv.PolyData(boundary_box) + p_cloud
            p_cloud_complete = pv.PolyData(boundary_box) + p_cloud_complete
    else:
        p_cloud = pv.PolyData(mesh[0].vertices[0])
        p_cloud_complete = pv.PolyData(mesh[0].vertices[0])

    original_model_points = []
    for i in range(len(original_model)):
        original_model_points.append(original_model[i].points)

    p_cloud_complete['label'] = np.array([i for i in range(p_cloud_complete.n_points)])
    p_cloud_complete['used_points'] = np.array([0 for i in range(p_cloud_complete.n_points)])
    n_points_p_cloud = p_cloud.n_points
    arr_for_filling = np.ones((p_cloud_complete.n_points - p_cloud.n_points, 3)) * 1000
    p_cloud.points = np.vstack((p_cloud.points, arr_for_filling))

    added_points = np.zeros(p_cloud.n_points)
    added_points[:n_points_p_cloud] = 1

    p_cloud['label'] = np.array([i for i in range(p_cloud.n_points)])
    p_cloud['used_points'] = np.array([0 for i in range(p_cloud.n_points)])
    p_cloud['added_points'] = added_points

    def change_point_colour(point_cloud, idx_vertices):
        for idx in idx_vertices:
            point_cloud['used_points'][idx] = 1

    def update_point_cloud(point_cloud, idx_vertices, pos_vertices):
        for i in range(len(idx_vertices)):
            if idx_vertices[i] >= len(point_cloud['added_points']):
                pass
            elif point_cloud['added_points'][idx_vertices[i]] == 0:
                point_cloud.points[idx_vertices[i]] = pos_vertices[i]
                point_cloud['added_points'][idx_vertices[i]] = 1

    def append_vertex_point_cloud(point_cloud, pos_vertex):
        point_cloud.points = np.vstack((point_cloud.points, pos_vertex))
        point_cloud['used_points'] = np.append(point_cloud['used_points'], 0)
        point_cloud['label'] = np.append(point_cloud['label'], (point_cloud.n_points - 1))

    def add_point_from_picking(point):
        picked_point_hashed = hash(frozenset(point))  # hash value of the picked point
        hashed_points = []  # list to store hash values of the points
        for ind_point in p_cloud_complete.points:  # iterate over the points of the cloud and calculate hash values
            hashed_points.append(hash(frozenset(ind_point)))  # hashing the points and appending

        if picked_point_hashed in hashed_points:  # test if the hash of the picked point is in the hash list of the pc
            # find corresponding idx of the picked face in the input mesh
            idx_point = np.nonzero(np.array(hashed_points) == picked_point_hashed)[0]
            compare_vert_to_picked = (point == p_cloud_complete.points[idx_point])
            if np.all(compare_vert_to_picked):
                update_point_cloud(p_cloud, idx_point, [point])
            else:
                append_vertex_point_cloud(p_cloud_complete, point)
                append_vertex_point_cloud(p_cloud, point)
        else:
            append_vertex_point_cloud(p_cloud_complete, point)
            append_vertex_point_cloud(p_cloud, point)

    def activate_vertex_picking(check):
        if not check:
            p.remove_actor('disableButton')
            p.add_text('enable vertices', position=(50, 10), font_size=10, color='black', name='enableButton')
            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()

            p.add_point_labels(label_meshes, "label", point_size=20, font_size=36, tolerance=0.01,
                               show_points=True, name='label_mesh')
        else:
            p.remove_actor('enableButton')
            p.add_text('disable vertices', position=(50, 10), font_size=10, color='black', name='disableButton')

            p.remove_actor('label_mesh-points')
            p.remove_actor('label_mesh-labels')

            p.remove_actor(text)
            p.add_text('Press middle button to set vertices for the control cage via console input.\n'
                       'Press right button to undo latest face.\n'
                       'Press left button to disable/activate vertex indices in the plot\n'
                       'Press third button to plot all boundary vertices\n'
                       'Press P under the mouse courser to pick additional points, for example inside the surfaces',
                       position='upper_left', name='textbox', color='black')

            p.add_checkbox_button_widget(pop_latest_face, position=(800, 10), color_on='red', color_off='red', size=35)
            p.add_text('Undo last face', position=(840, 10), font_size=10, color='black')

            points.append(p.add_mesh(p_cloud, point_size=20, scalars='used_points',
                                     cmap=['white', 'black'], show_scalar_bar=False, pickable=False))
            labels.append(p.add_point_labels(p_cloud, "label", point_size=20, font_size=36, tolerance=0.01,
                                             show_points=False))

            p.add_checkbox_button_widget(select_vertices, position=(300, 10), size=35, color_on='green',
                                         color_off='green')
            p.add_text('set vertices', position=(345, 10), font_size=10, color='black')
            p.add_checkbox_button_widget(show_boundary_idx, position=(480, 10), size=35, color_on="purple")
            p.add_text('enable boundary vertices', position=(525, 10), font_size=10, color='black', name='boundary')

    def show_boundary_idx(check):
        if 'enableButton' in p.renderer.actors:
            return
        if not check:
            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()
            p.add_text('enable boundary vertices', position=(525, 10), font_size=10, color='black', name='boundary')
            activate_vertex_picking(True)
        else:

            p.remove_actor(labels)
            p.remove_actor(points)
            points.pop()
            labels.pop()
            points.append(p.add_mesh(p_cloud_complete, point_size=20, scalars='used_points',
                                     cmap=['white', 'black'], show_scalar_bar=False, pickable=False))
            labels.append(p.add_point_labels(p_cloud_complete, "label", point_size=20, font_size=36, tolerance=0.01,
                                             show_points=False))
            p.add_text('disable boundary vertices', position=(525, 10), font_size=10, color='black', name='boundary')

    def pop_latest_face(check):
        if check:
            if len(control_cage_faces) == 0:
                print('no faces defined for control cage')
            else:
                print('saved faces: ', control_cage_faces)
                print('deleted face: ', control_cage_faces[-1])
                added_triangles.discard(tuple(sorted(control_cage_faces.pop())))
                for i in range(3):
                    p.remove_actor(line[-1])
                    line.pop()
                p.remove_actor(faces[-1])
                faces.pop()
                dynamic_face.pop()
                vertices_fit.pop()
                selected_points.clear()
                print('saved faces: ', control_cage_faces)
        else:
            pop_latest_face(True)

    def select_vertices(check):
        if check:
            print('choose three control vertices to form a triangular face')
            while True:
                input_0 = input('vertex one: ') or False
                if not input_0:
                    return
                else:
                    try:
                        selected_points.append(int(input_0))
                        break
                    except ValueError:
                        print(f"The input {input_0} cannot be interpreted as an integer, please try again or "
                              f"leave blank to cancel!")
            while True:
                input_1 = input('vertex two: ') or False
                if not input_1:
                    selected_points.clear()
                    return
                else:
                    try:
                        input_1_int = int(input_1)
                        if input_1_int in selected_points:
                            print(f"the input {input_1} already in selected points, please try again or leave blank"
                                  f"to cancel")
                        else:
                            selected_points.append(input_1_int)
                            break
                    except ValueError:
                        print(f"The input {input_1} cannot be interpreted as an integer, please try again or "
                              f"leave blank to cancel!")
            while True:
                input_2 = input('vertex three: ') or False
                if not input_2:
                    selected_points.clear()
                    return
                else:
                    try:
                        input_2_int = int(input_2)
                        if input_2_int in selected_points:
                            print(f"the input {input_2} already in selected points, please try again or leave blank "
                                  f"to cancel")
                        else:
                            selected_points.append(input_2_int)
                            break
                    except ValueError:
                        print(f"The input {input_2} cannot be interpreted as an integer, please try again or "
                              f"leave blank to cancel!")

            vertices = []  # list to store the coordinates of the three selected vertices
            vertices_sorted = tuple(sorted(selected_points))  # temp tuple for sorted vertices for check of dublicates
            if vertices_sorted in added_triangles:
                print(f"Triangle {selected_points} already picked please make different triangle")
                selected_points.clear()
            else:
                added_triangles.add(tuple(sorted(selected_points)))

                for vert_idx in selected_points:  # iterate over the selected vertices and append to list
                    vertices.append(p_cloud_complete.points[vert_idx].tolist())

                if use_dynamic_faces:
                    # set mesh to fit the selected vertices to
                    mesh_id, verts_mesh = automatization.find_mesh_for_fitting(vertices, mesh)
                    saved_actor = p.renderer.actors['mesh_' + str(mesh_id)]
                    p.remove_actor('mesh_' + str(mesh_id))
                    p.add_mesh(original_model[mesh_id], color='red', use_transparency=False, style='surface',
                               show_edges=True,
                               name='visualisation', pickable=False)

                    while True:
                        conf_mesh = input(f'mesh_{mesh_id} selected for fitting. \n'
                                          f'For conformation please press Enter, or input different mesh index.\n'
                                          f'To set triangle as static write "static" or "s",\n'
                                          f'to cancel write "cancel" or "c": ') or 'y'
                        if conf_mesh == 'y':
                            dynamic_face.append(np.array(str(mesh_id)))
                            vertices_fit.append(np.array(verts_mesh))
                            break

                        elif conf_mesh[0].lower() == 'c':
                            p.add_actor(saved_actor, name='mesh_' + str(mesh_id))
                            p.remove_actor('visualisation')
                            added_triangles.discard(tuple(sorted(selected_points)))
                            selected_points.clear()
                            return

                        elif conf_mesh[0].lower() == 's':
                            dynamic_face.append('s')

                            if add_boundary_box:
                                for i in range(len(selected_points)):
                                    vertex = tuple(vertices[i])
                                    mesh_vertices = set()
                                    for vert in mesh[verts_mesh[i]].vertices:
                                        mesh_vertices.add(tuple(vert))
                                    if vertex not in mesh_vertices:
                                        verts_mesh[i] = 's'
                            vertices_fit.append(np.array(verts_mesh))
                            break
                        else:
                            try:
                                corr_mesh = int(conf_mesh)
                                if corr_mesh > len(mesh) - 1:
                                    print(f"the input index {corr_mesh} is out range for indices of the given meshes, "
                                          f"only {len(mesh)} mesh(es) given as input with a maximal "
                                          f"index of {len(mesh) - 1}")
                                else:
                                    dynamic_face.append(np.array(conf_mesh))
                                    vertices_fit.append(np.ones(len(verts_mesh)) * corr_mesh)
                                    break
                            except ValueError:
                                print(f"The input {conf_mesh} cannot be interpreted, please try again")

                    p.add_actor(saved_actor, name='mesh_' + str(mesh_id))
                    p.remove_actor('visualisation')

                print('selected vertices: ', selected_points)

                update_point_cloud(p_cloud, selected_points, vertices)
                change_point_colour(p_cloud, selected_points)
                change_point_colour(p_cloud_complete, selected_points)

                # selected_points_sorted = np.sort(selected_points)
                selected_points_sorted = selected_points
                face_array = [3, 0, 1, 2]
                faces_for_visual = pv.PolyData(p_cloud_complete.points[selected_points], face_array)
                face_actor = p.add_mesh(faces_for_visual, color='green', pickable=False)
                faces.append(face_actor)
                line1 = pv.Line(p_cloud_complete.points[selected_points_sorted[0]],
                                p_cloud_complete.points[selected_points_sorted[1]])
                line2 = pv.Line(p_cloud_complete.points[selected_points_sorted[1]],
                                p_cloud_complete.points[selected_points_sorted[2]])
                line3 = pv.Line(p_cloud_complete.points[selected_points_sorted[2]],
                                p_cloud_complete.points[selected_points_sorted[0]])
                line1_actor = p.add_mesh(line1, color="blue", line_width=5, pickable=False)
                line2_actor = p.add_mesh(line2, color="blue", line_width=5, pickable=False)
                line3_actor = p.add_mesh(line3, color="blue", line_width=5, pickable=False)
                line.append(line1_actor)
                line.append(line2_actor)
                line.append(line3_actor)
                control_cage_faces.append(np.array(selected_points_sorted))
                print('saved faces: ', control_cage_faces)
                selected_points.clear()
        else:
            select_vertices(True)

    def make_mesh_invisible(check):
        if check:
            mesh_nr = int(input('Please enter number of mesh: '))
            if mesh_nr > len(original_model) - 1:
                print(f'Index {mesh_nr} is out of range, try again.')
            else:
                if mesh_nr in invisible_meshes:
                    print(f'mesh_{mesh_nr} already invisible')
                else:
                    p.add_mesh(original_model[mesh_nr], color=color, opacity=0.0, style=render_style, show_edges=True,
                               name=f'mesh_{mesh_nr}', pickable=False)
                    invisible_meshes.append(mesh_nr)
        else:
            make_mesh_invisible(True)

    def make_mesh_visible(check):
        if check:
            if len(invisible_meshes) == 0:
                print('All meshes are visible')
            else:
                for invisible_mesh in invisible_meshes:
                    print(invisible_mesh)
                    p.add_mesh(original_model[invisible_mesh], color=color, use_transparency=False, style=render_style,
                               show_edges=True, name=f'mesh_{invisible_mesh}')
            invisible_meshes.clear()
        else:
            make_mesh_visible(True)

    for model_nr, model in enumerate(original_model):
        name = 'mesh_' + str(model_nr)
        p.add_mesh(model, color=color, use_transparency=False, style=render_style, show_edges=True, name=name)

    p.enable_point_picking(add_point_from_picking, show_message=False)

    p.add_checkbox_button_widget(make_mesh_invisible, position=(10, 130), color_on='yellow',
                                 color_off='yellow', size=35)
    p.add_text('make mesh \n invisible', position=(10, 85), font_size=10, color='black')
    p.add_checkbox_button_widget(make_mesh_visible, position=(10, 230), color_on='orange',
                                 color_off='orange', size=35)

    p.add_text('make mesh \n visible', position=(10, 185), font_size=10, color='black')

    p.add_checkbox_button_widget(activate_vertex_picking, position=(10, 10), size=35)

    p.isometric_view_interactive()
    text = p.add_text('Press button to select control vertices', position='lower_edge', name='textbox')
    p.set_background('royalblue', top='aliceblue')
    p.show()

    vertices_position = []
    control_cage_faces_flattend = np.array(control_cage_faces).flatten()
    control_cage_unique_idx_sorted, indices, inverse = np.unique(control_cage_faces, return_index=True,
                                                                 return_inverse=True)

    control_cage_faces_unique_idx = np.asanyarray([control_cage_faces_flattend[index] for index in sorted(indices)])
    new_idx = []

    for i in range(len(control_cage_faces)):
        for idx in control_cage_faces[i]:
            new_idx.append(np.nonzero(control_cage_faces_unique_idx == idx)[0][0])

    for idx in control_cage_faces_unique_idx:
        vertices_position.append(p_cloud_complete.points[idx].tolist())

    dynamic_face_flattened = np.array(dynamic_face, dtype=str).flatten()
    control_cage_faces = new_idx
    control_cage_faces_reshaped = np.reshape(new_idx, (-1, 3))

    face_index = 0
    for face in control_cage_faces_reshaped:
        vert_index = 0
        for vert in face:
            dynamic_vertices_dict[vert] = vertices_fit[face_index][vert_index]
            vert_index += 1
        face_index += 1
    dynamic_vertices = np.zeros(len(dynamic_vertices_dict), dtype=str)
    for vertex in dynamic_vertices_dict:
        if not isinstance(dynamic_vertices_dict[vertex], list):
            dynamic_vertices[vertex] = str(dynamic_vertices_dict[vertex])
        else:
            unique_indices, counts = np.unique(dynamic_vertices_dict[vertex], return_counts=True)
            if sum(counts) == len(unique_indices):  # all meshes have the same occurrence take the first one
                dynamic_vertices[vertex] = str(dynamic_vertices_dict[vertex][0])
            else:  # take the mesh with maximal occurence
                index_max_occurrence = np.argmax(counts)
                dynamic_vertices[vertex] = str(dynamic_vertices_dict[vertex][index_max_occurrence])

    control_cage = PySubdiv.Mesh(vertices_position, control_cage_faces_reshaped)
    if use_dynamic_faces:
        control_cage.data['fitting_method'] = 'dynamic_faces'
    else:
        control_cage.data['fitting_method'] = None
    control_cage.data['dynamic_faces'] = dynamic_face_flattened
    control_cage.data['dynamic_vertices'] = dynamic_vertices

    if simple_subdivision:
        control_cage = control_cage.simple_subdivision(iteration_subdivision)

    return control_cage


class ControlMesh:

    def __init__(self, input_meshes):
        self.input_meshes = self.check_meshes(input_meshes)
        self.plotter = pv.Plotter()  # PyVista.Plotter
        self.selection = []  # empty list to store selection of vertices
        self.selection_idx = []  # empty list to store idx of selected vertices
        self.selected_meshes = []  # list to store selection of meshes
        self.invisible_meshes = []  # keep track of invisible meshes
        self.button_widgets_invisible = []  # button widgets of meshes
        self.button_widgets = {}
        self.button_widgets_boundary = {}  # index of button widget in plotter.button_widgets
        self.button_widgets_utils = {}  # button widget and index in plotter.button_widgets
        # self.selection_polygon = []  # list to store selection for polygon cutting
        self.selection_polygon_idx = []  # list for indices for cutting polygon
        self.selection_polygon_mesh = pv.PolyData()
        self.selection_mesh = pv.PolyData()
        self.new_meshes_idx = {}  # key index new mesh -> value index in self.input_meshes
        self.merged_meshes_idx = {}  # key index merged mesh -> value index in self.input_meshes
        self.dynamic_vertices = {}
        self.decimated_mesh = pv.PolyData()  # field to hold the decimated mesh
        self._state = None

    def check_meshes(self, input_meshes):
        list_poly_data = []
        # check if meshes are in a list
        if isinstance(input_meshes, list):
            # check if meshes are PySubdiv.Mesh or PyVistaPolydata
            for idx, mesh in enumerate(input_meshes):
                if isinstance(mesh, (PySubdiv.Mesh, pyvista.core.pointset.PolyData)):
                    # check which type, if PySubdiv.Mesh convert to PyVista.PolyData and append
                    if isinstance(mesh, PySubdiv.Mesh):
                        list_poly_data.append(mesh.model().clean())
                    # if not append directly
                    else:
                        list_poly_data.append(mesh.clean())
                else:
                    raise TypeError(f"mesh at index {idx} is type {type(mesh)}. "
                                    f"Types should be PySubdiv.Mesh or Pyvista.PolyData")
        # check correct type of passed single mesh
        else:
            if isinstance(input_meshes, (PySubdiv.Mesh, pyvista.core.pointset.PolyData)):
                if isinstance(input_meshes, (PySubdiv.Mesh, pyvista.core.pointset.PolyData)):
                    # check which type, if PySubdiv.Mesh convert to PyVista.PolyData and append
                    if isinstance(input_meshes, PySubdiv.Mesh):
                        list_poly_data.append(input_meshes.model().clean())
                    # if not append directly
                    else:
                        list_poly_data.append(input_meshes.clean())
            else:
                raise TypeError(f"mesh is type {type(input_meshes)}. "
                                f"Types should be PySubdiv.Mesh or Pyvista.PolyData")
        self.input_meshes = list_poly_data
        return self.input_meshes

    def mesh_visibility(self, check, idx_mesh=None):
        button_idx = self.button_widgets[f"button_invisible_{idx_mesh}"]
        if check:
            # check if mesh is selected; if selected set state of the button to False
            if idx_mesh in self.selected_meshes:
                print(f"mesh_{idx_mesh} is selected, please deselect first")
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)

            # set mesh to invisible
            else:
                self.invisible_meshes.append(idx_mesh)
                self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                      name=f"mesh_{idx_mesh}", opacity=0)
                self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-points')
                self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-labels')
        # make mesh visible
        else:
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100)
            self.plotter.add_point_labels([self.input_meshes[idx_mesh].center_of_mass()], [f"mesh_{idx_mesh}"],
                                          name=f"labels_mesh_{idx_mesh}")
            self.invisible_meshes.remove(idx_mesh)

    def select_mesh(self, check, idx_mesh=None):
        button_idx = self.button_widgets[f"button_{idx_mesh}"]
        if check:
            if self._state != States.picking_selection:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                print('Selection is not possible')
                return

            # check if mesh is invisible; if invisible set state of button to false
            if idx_mesh in self.invisible_meshes:
                print(f"mesh_{idx_mesh} is invisible and cannot be selected")
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
            # select the current mesh
            else:
                self.selected_meshes.append(idx_mesh)
                self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                      name=f"mesh_{idx_mesh}", opacity=100)
                # set button to constrain the selection only on the boundary vertices/min max points

                button_boundary_idx = self.button_widgets_boundary[f"button_boundary_{idx_mesh}"]
                self.plotter.button_widgets[button_boundary_idx].On()
                self.plotter.button_widgets[button_boundary_idx].GetRepresentation().VisibilityOn()

                button_min_max_idx = self.button_widgets[f"button_min_max_{idx_mesh}"]
                self.plotter.button_widgets[button_min_max_idx].On()
                self.plotter.button_widgets[button_min_max_idx].GetRepresentation().VisibilityOn()
        # deselect the mesh
        else:

            # check if boundary vertices actor is active -> remove
            if f"constrained_mesh{idx_mesh}" in self.plotter.renderer.actors:
                self.plotter.remove_actor(f"constrained_mesh{idx_mesh}")

            # deactivate the button for boundary vertices/min max when the mesh is deselected
            button_boundary_idx = self.button_widgets_boundary[f"button_boundary_{idx_mesh}"]
            self.plotter.button_widgets[button_boundary_idx].GetRepresentation().SetState(False)
            self.plotter.button_widgets[button_boundary_idx].GetRepresentation().VisibilityOff()

            button_min_max_idx = self.button_widgets[f"button_min_max_{idx_mesh}"]
            self.plotter.button_widgets[button_min_max_idx].GetRepresentation().SetState(False)
            self.plotter.button_widgets[button_min_max_idx].GetRepresentation().VisibilityOff()

            # make mesh unpickable and deselect
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100)
            # set state of the button to false
            self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
            self.selected_meshes.remove(idx_mesh)

    def point_to_selection(self, mesh, point_idx):
        if self._state == States.picking_selection:
            # Get coordinates of the point
            point = mesh.points[point_idx]
            # Append to selection
            self.selection.append(point)
            self.selection_mesh += pv.PolyData(point)
            self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10, name='selection')
        elif self._state == States.deleting_selection:
            self.selection.pop(point_idx)
            if len(self.selection) == 0:
                self.selection_mesh = pv.PolyData()
                self.plotter.remove_actor('selection')

                # if all points deleted set state back to picking
                self.switch_picking_deleting(True)
                button_idx = self.button_widgets_utils['button_picking_deleting']
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(True)
                return

            self.selection_mesh = pv.PolyData(self.selection)
            self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=True, point_size=10, name='selection')

        else:
            return

    def point_to_polygon(self, mesh, point_idx):
        if point_idx in self.selection_polygon_idx:
            return
        self.selection_polygon_idx.append(point_idx)
        self.selection_polygon_mesh.faces = np.array([len(self.selection_polygon_idx), ] + self.selection_polygon_idx)
        if self.selection_polygon_mesh.faces[0] > 2:
            self.selection_polygon_mesh = self.selection_polygon_mesh.compute_normals(cell_normals=True,
                                                                                      point_normals=False)
            # normal of polygon face
            normal = self.selection_polygon_mesh["Normals"][0]
            # center point of the polygon
            center = self.selection_polygon_mesh.cell_centers().points[0]
            # dot product from normal and vector to current camera position
            dot_product = np.dot(normal, (center - self.plotter.camera_position[0]))
            if dot_product > 0:
                color = 'lightblue'
            else:
                color = 'yellow'

            self.plotter.add_mesh(self.selection_polygon_mesh, color=color, pickable=False, point_size=15,
                                  name='selection_polygon', show_edges=True)

    def triangulate_selection(self, check):
        if self._state == States.mesh_decimation:
            print('Mesh decimation activated')
            return
        if check:
            # check if vertices are selected
            if len(self.selection) < 3:
                print('Less than 3 vertices in the selection')
                return

            # user_input = input("Polygon for cutting? ") or False

            msg = 'Use polygon for cutting?'
            user_input = easygui.ynbox(msg)

            if user_input:
                if "polygon_button" in self.button_widgets_utils:
                    idx_polygon_button = self.button_widgets_utils['polygon_button']
                    self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetVisibility(True)
                    self.plotter.button_widgets[idx_polygon_button].On()
                else:
                    self.plotter.add_checkbox_button_widget(self.selection_to_polygon, position=(80, 10))
                    self.button_widgets_utils['polygon_button'] = len(self.plotter.button_widgets) - 1
                self.plotter.add_text('abort', font_size=6, position=(16, 70), name="triangulate_abort_text")

            else:
                if len(self.selection) == 3:
                    mesh = pv.PolyData(np.array(self.selection), faces=[3, 0, 1, 2])
                else:
                    mesh = pv.PolyData(np.array(self.selection)).delaunay_2d(edge_source=self.selection_polygon_mesh)
                idx_mesh = len(self.input_meshes)  # index in self.input_meshes
                # add the mesh to the plotter
                self.add_meshes_to_input_and_plotter(mesh, idx_mesh, new_mesh=True)
                button_idx = self.button_widgets_utils['triangulation_button']
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
        else:
            if self._state == States.polygon_cutting:
                self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, pickable_window=False)
                self.selection_polygon_mesh = pv.PolyData()
                self.plotter.remove_actor('selection_polygon')
                self.selection_polygon_idx.clear()
                self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                      name='selection')

                idx_polygon_button = self.button_widgets_utils['polygon_button']
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetVisibility(False)
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetState(False)
                self.plotter.button_widgets[idx_polygon_button].Off()
                self.plotter.remove_actor("triangulate_abort_text")
                self.set_state(States.picking_selection)
            else:
                button_idx = self.button_widgets_utils['triangulation_button']
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)

                idx_polygon_button = self.button_widgets_utils['polygon_button']
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetVisibility(False)
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetState(False)
                self.plotter.button_widgets[idx_polygon_button].Off()
                self.plotter.remove_actor("triangulate_abort_text")

    def constrain_selection_to_boundary(self, check, idx_mesh):
        if check:

            # uncheck button for min max vertices
            button_min_max_idx = self.button_widgets[f"button_min_max_{idx_mesh}"]
            self.plotter.button_widgets[button_min_max_idx].GetRepresentation().SetState(False)

            # extract vertices on the boundary and make mesh unpickable only points on the boundary are pickable
            mesh = self.input_meshes[idx_mesh]
            boundary = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                  manifold_edges=False, feature_edges=False)
            # threshold = float(input('Enter threshold angle: ')) or 0
            msg = 'Enter threshold angle between edges (float): '
            successful_input = False
            while not successful_input:
                threshold = easygui.enterbox(msg, default='0')
                if threshold is None:
                    threshold = 0
                    successful_input = True
                else:
                    try:
                        threshold = float(threshold)
                        successful_input = True
                    except ValueError:
                        print("Input is not a float, please try again")

            connected_vertices = {vertex_idx: [] for vertex_idx in range(boundary.n_points)}
            counter = 2
            for line_idx in range(boundary.n_lines):
                line_vertex_idx_1 = boundary.lines[counter - 1]
                line_vertex_idx_2 = boundary.lines[counter]
                connected_vertices[line_vertex_idx_1].append(line_vertex_idx_2)
                connected_vertices[line_vertex_idx_2].append(line_vertex_idx_1)
                counter += 3  # two vertices per edge

            vertices_after_threshold = []
            for idx_vertex in range(boundary.n_points):
                point_1, point_2 = boundary.points[connected_vertices[idx_vertex]]
                footpoint = boundary.points[idx_vertex]
                angle = calculation.angle_vertices(point_1, point_2, footpoint)
                angle = abs((angle * 180 / np.pi) - 180)
                if angle > threshold:
                    vertices_after_threshold.append(footpoint)
            boundary_vertices = np.array(vertices_after_threshold)
            self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100)

            self.plotter.add_points(boundary_vertices, pickable=True, name=f"constrained_mesh{idx_mesh}")

        else:
            # remove actor for vertices on the boundary and make the mesh pickable again
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100)
            self.plotter.remove_actor(f"constrained_mesh{idx_mesh}")

    def selection_to_polygon(self, check):
        if check:
            self.deselect_all_meshes(True)
            self._state = States.polygon_cutting
            self.plotter.enable_point_picking(self.point_to_polygon, use_mesh=True, pickable_window=False)
            self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=True, point_size=10, name='selection')
            self.selection_polygon_mesh.points = np.array(self.selection)
        else:
            # delauny triangulation with the polygon as mask
            mesh = pv.PolyData(np.array(self.selection)).delaunay_2d(edge_source=self.selection_polygon_mesh)
            idx_mesh = len(self.input_meshes)  # index in self.input_meshes
            # self.new_meshes_idx[idx_mesh] = len(self.new_meshes_idx.keys())

            # add the mesh to the plotter
            self.add_meshes_to_input_and_plotter(mesh, idx_mesh, new_mesh=True)

            # remove the selection polygon from plotter + clear lists
            self.plotter.remove_actor('selection_polygon')
            self.selection_polygon_idx.clear()
            self.selection_polygon_mesh = pv.PolyData()

            # deactivate the button end enable picking on meshes
            self.plotter.button_widgets[self.button_widgets_utils["polygon_button"]].Off()
            self.plotter.button_widgets[self.button_widgets_utils["polygon_button"]].GetRepresentation().SetState(False)
            self.plotter.button_widgets[
                self.button_widgets_utils["polygon_button"]].GetRepresentation().SetVisibility(False)

            # self.button_widgets_utils.pop("polygon_button", None)
            self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, pickable_window=False)
            self._state = States.picking_selection

    def selection_undo(self, check):
        if check:
            if self._state == States.picking_selection:
                if len(self.selection) == 0:
                    return
                self.selection.pop()
                if len(self.selection) == 0:
                    self.selection_mesh = pv.PolyData()
                    self.plotter.remove_actor('selection')
                    return

                self.selection_mesh = pv.PolyData(np.array(self.selection))
                self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                      name='selection')

            elif self._state == States.polygon_cutting:
                if len(self.selection_polygon_idx) == 0:
                    return

                self.selection_polygon_idx.pop()
                self.selection_polygon_mesh.faces = np.array([len(self.selection_polygon_idx), ]
                                                             + self.selection_polygon_idx)

                self.plotter.add_mesh(self.selection_polygon_mesh, color='lightblue', pickable=False, point_size=15,
                                      name='selection_polygon', show_edges=True)
            else:
                return
        else:
            self.selection_undo(True)

    def switch_picking_deleting(self, picking):
        button_idx = self.button_widgets_utils['button_picking_deleting']
        button_state = self.plotter.button_widgets[button_idx].GetRepresentation().GetState()
        if self._state != States.picking_selection and self._state != States.deleting_selection:
            if button_state:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
            else:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(True)
            return
        if picking:
            self.set_state(States.picking_selection)
            self.plotter.add_text('select', font_size=6, position=(210, 60), name="selection_text")
            if len(self.selection) != 0:
                self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                      name='selection')

        else:
            if len(self.selection) == 0:
                if button_state:
                    self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                else:
                    self.plotter.button_widgets[button_idx].GetRepresentation().SetState(True)
                return

            self.set_state(States.deleting_selection)
            self.plotter.add_text('delete', font_size=6, position=(210, 60), name="selection_text")
            self.deselect_all_meshes(True)
            self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=True, point_size=10, name='selection')

    def add_meshes_to_input_and_plotter(self, mesh, idx_mesh, new_mesh=False, merged_mesh=False):
        if None in self.input_meshes:
            idx_mesh = self.input_meshes.index(None)
            self.input_meshes[idx_mesh] = mesh

        else:
            self.input_meshes.append(mesh)

        if new_mesh:
            if None in self.new_meshes_idx.values():
                counter = 0
                for mesh_index, new_mesh_index in self.new_meshes_idx.items():
                    if new_mesh_index is None:
                        # search for the next free slot
                        while counter in self.new_meshes_idx.values():
                            counter += 1
                        self.new_meshes_idx[idx_mesh] = counter
                        break
                    else:
                        counter += 1
            else:
                self.new_meshes_idx[idx_mesh] = len(self.new_meshes_idx.keys())

            # find boundary_points of the new mesh
            boundary = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                  manifold_edges=False, feature_edges=False)
            boundary_points = boundary.points

            edges = mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

            boundary_edges = boundary.lines.reshape((-1, 3))[:, 1:]
            # points_masked_non_manifold_edges = non_manifold_points[non_manifold_edges]
            # print(points_masked_non_manifold_edges)
            # print(points_masked_edges)

            edge_dict = {}

            for edge_idx, edge in enumerate(boundary_edges):
                edge_masked = boundary_points[edge]
                if tuple(edge_masked[0].tolist()) not in edge_dict:
                    edge_dict[tuple(edge_masked[0].tolist())] = {tuple(edge_masked[1].tolist())}
                else:
                    edge_dict[tuple(edge_masked[0].tolist())].update([tuple(edge_masked[1].tolist())])
                if tuple(edge_masked[1].tolist()) not in edge_dict:
                    edge_dict[tuple(edge_masked[1].tolist())] = {tuple(edge_masked[0].tolist())}
                else:
                    edge_dict[tuple(edge_masked[1].tolist())].update([tuple(edge_masked[0].tolist())])
            boundary_edges_data = np.zeros(len(edges))
            for edge_idx, edge in enumerate(edges):
                edge_masked = mesh.points[edge]
                if tuple(edge_masked[0].tolist()) in edge_dict:
                    if tuple(edge_masked[1].tolist()) in edge_dict[tuple(edge_masked[0].tolist())]:
                        boundary_edges_data[edge_idx] = 1
            vertex_dict = {}
            boundary_vertices = np.ones(mesh.n_points, dtype=int)
            for point_idx, point in enumerate(boundary_points):
                vertex_dict[tuple(point.tolist())] = point_idx
            for vrtx_idx, vrtx in enumerate(mesh.points):
                if tuple(vrtx.tolist()) not in vertex_dict:
                    boundary_vertices[vrtx_idx] = -1

            mesh.point_data['boundary_vertices'] = boundary_vertices
            mesh.field_data['boundary_edges_data'] = boundary_edges_data

            # set dynamic_vertices for the new mesh

            msg = 'Enter mesh index for dynamic vertices or cancel'
            dynamic_vertex = easygui.enterbox(msg)
            if dynamic_vertex is not None:
                dynamic_vertices = np.array([dynamic_vertex for _ in range(mesh.n_points)], dtype=str)
                dynamic_faces = np.array([dynamic_vertex for _ in range(mesh.n_cells)], dtype=str)
                mesh["dynamic_vertices"] = dynamic_vertices
                mesh["dynamic_faces"] = dynamic_faces

        if merged_mesh:
            if None in self.merged_meshes_idx.values():
                print(self.merged_meshes_idx)
                counter = 0
                for mesh_index, merged_mesh_index in self.merged_meshes_idx.items():
                    if merged_mesh_index is None:
                        # search for the next free slot
                        while counter in self.merged_meshes_idx.values():
                            counter += 1
                        self.merged_meshes_idx[idx_mesh] = counter
                        break
                    else:
                        counter += 1
            else:
                self.merged_meshes_idx[idx_mesh] = len(self.merged_meshes_idx.keys())

            # find non-manifold vertices of the merged mesh
            non_manifold_edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
                                                            manifold_edges=False, feature_edges=False)

            non_manifold_points = non_manifold_edges.points

            vertex_dict = {}
            non_manifold_vertices = np.ones(mesh.n_points, dtype=int)
            for point_idx, point in enumerate(non_manifold_points):
                vertex_dict[tuple(point.tolist())] = point_idx
            for vrtx_idx, vrtx in enumerate(mesh.points):
                if tuple(vrtx.tolist()) not in vertex_dict:
                    non_manifold_vertices[vrtx_idx] = -1

            mesh.point_data['non_manifold_vertices'] = non_manifold_vertices

        # add the mesh to the plotter
        self.plotter.add_mesh(mesh, color="green", show_edges=True, name=f"mesh_{idx_mesh}", pickable=False)
        self.plotter.add_point_labels([mesh.center_of_mass()], [f"mesh_{idx_mesh}"], name=f"labels_mesh_{idx_mesh}")
        x_pos = 10
        offset = 50 * idx_mesh
        if new_mesh:
            x_pos = 115
            offset = 50 * self.new_meshes_idx[idx_mesh]
        if merged_mesh:
            x_pos = 220
            offset = 50 * self.merged_meshes_idx[idx_mesh]
        self.plotter.add_text(f"mesh_{idx_mesh}", position=(x_pos, 700 - offset), font_size=10,
                              name=f"mesh_name_{idx_mesh}")
        # add button widgets

        vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.mesh_visibility, idx_mesh=idx_mesh,
                                                 position=(x_pos, 680 - offset), size=15, border_size=2)

        self.button_widgets[f"button_invisible_{idx_mesh}"] = len(self.plotter.button_widgets) - 1

        vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.select_mesh, idx_mesh=idx_mesh,
                                                 position=(x_pos + 20, 680 - offset), size=15, border_size=2,
                                                 color_on='red')

        self.button_widgets[f"button_{idx_mesh}"] = len(self.plotter.button_widgets) - 1

        vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.constrain_selection_to_boundary,
                                                 idx_mesh=idx_mesh, position=(x_pos + 40, 680 - offset), size=15,
                                                 border_size=2)

        # save the index of the button in the self.plotter.button_widgets property for deletion
        self.button_widgets_boundary[f"button_boundary_{idx_mesh}"] = len(self.plotter.button_widgets) - 1

        button_boundary_idx = self.button_widgets_boundary[f"button_boundary_{idx_mesh}"]
        # set state and turn visibility of button off
        self.plotter.button_widgets[button_boundary_idx].Off()
        self.plotter.button_widgets[button_boundary_idx].GetRepresentation().SetState(False)
        self.plotter.button_widgets[button_boundary_idx].GetRepresentation().VisibilityOff()

        vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.constrain_selection_min_max,
                                                 idx_mesh=idx_mesh, position=(x_pos + 60, 680 - offset), size=15,
                                                 border_size=2)

        # save the index of the button in the self.plotter.button_widgets property for deletion
        self.button_widgets[f"button_min_max_{idx_mesh}"] = len(self.plotter.button_widgets) - 1

        button_min_max_idx = self.button_widgets[f"button_min_max_{idx_mesh}"]
        # set state and turn visibility of button off
        self.plotter.button_widgets[button_min_max_idx].Off()
        self.plotter.button_widgets[button_min_max_idx].GetRepresentation().SetState(False)
        self.plotter.button_widgets[button_min_max_idx].GetRepresentation().VisibilityOff()

        vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.delete_mesh, idx_mesh=idx_mesh,
                                                 position=(x_pos + 80, 680 - offset), size=15, border_size=2,
                                                 color_on='red')

        self.button_widgets[f"button_deletion{idx_mesh}"] = len(self.plotter.button_widgets) - 1
        return idx_mesh

    def deselect_all_meshes(self, check):
        # loop over selected meshes and deselect all
        if check:
            temp_list = copy.copy(self.selected_meshes)
            for idx_mesh in temp_list:
                self.select_mesh(False, idx_mesh)
        else:
            self.deselect_all_meshes(True)

    def clear_selection(self, check):
        if check:
            if self._state == States.picking_selection or self._state == States.deleting_selection:
                if len(self.selection) != 0:
                    self.selection.clear()
                    # self.selection_idx.clear()
                    self.selection_mesh = pv.PolyData()
                    self.plotter.remove_actor('selection')
                    button_index = self.button_widgets_utils['button_picking_deleting']
                    self.plotter.button_widgets[button_index].GetRepresentation().SetState(True)
                    self.set_state(States.picking_selection)

            if self._state == States.polygon_cutting:
                self.selection_polygon_idx.clear()
                self.selection_polygon_mesh = pv.PolyData()
                self.selection_polygon_mesh.points = np.array(self.selection)
                self.plotter.remove_actor('selection_polygon')

        else:
            self.clear_selection(True)

    def merge_meshes(self, check):
        if check:
            mesh_1 = self.input_meshes[int(input("Index first mesh: "))]
            mesh_2 = self.input_meshes[int(input("Index second mesh: "))]
            print(mesh_1.array_names)
            print(mesh_2.array_names)

            # self.merged_meshes_idx[idx_mesh] = len(self.merged_meshes_idx.keys())

            msg = 'Recalculate boundary vertices? '
            user_bool = easygui.ynbox(msg)
            if user_bool:
                # recalculate boundary_points of the new mesh
                merged_mesh = mesh_1.merge(mesh_2, inplace=False)
                idx_mesh = len(self.input_meshes)  # index in self.input_meshes

                boundary = merged_mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                             manifold_edges=False, feature_edges=False)
                boundary_points = boundary.points
                boundary_points_mesh_1 = mesh_1.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                      manifold_edges=False, feature_edges=False).points

                boundary_points_mesh_2 = mesh_2.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                      manifold_edges=False, feature_edges=False).points

                edges = merged_mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

                boundary_edges = boundary.lines.reshape((-1, 3))[:, 1:]
                # points_masked_non_manifold_edges = non_manifold_points[non_manifold_edges]
                # print(points_masked_non_manifold_edges)
                # print(points_masked_edges)

                edge_dict = {}

                for edge_idx, edge in enumerate(boundary_edges):
                    edge_masked = boundary_points[edge]
                    if tuple(edge_masked[0].tolist()) not in edge_dict:
                        edge_dict[tuple(edge_masked[0].tolist())] = {tuple(edge_masked[1].tolist())}
                    else:
                        edge_dict[tuple(edge_masked[0].tolist())].update([tuple(edge_masked[1].tolist())])
                    if tuple(edge_masked[1].tolist()) not in edge_dict:
                        edge_dict[tuple(edge_masked[1].tolist())] = {tuple(edge_masked[0].tolist())}
                    else:
                        edge_dict[tuple(edge_masked[1].tolist())].update([tuple(edge_masked[0].tolist())])
                boundary_edges_data = np.zeros(len(edges))
                for edge_idx, edge in enumerate(edges):
                    edge_masked = merged_mesh.points[edge]
                    if tuple(edge_masked[0].tolist()) in edge_dict:
                        if tuple(edge_masked[1].tolist()) in edge_dict[tuple(edge_masked[0].tolist())]:
                            boundary_edges_data[edge_idx] = 1

                merged_mesh.field_data['boundary_edges_data'] = boundary_edges_data

                if 'boundary_layer_vertices' in mesh_1.array_names:
                    print('2')
                    boundary_layers_old = mesh_1.points[np.nonzero(mesh_1['boundary_layer_vertices'] == 1)]
                    boundary_points_mesh_1 = np.concatenate((boundary_points_mesh_1, np.array(boundary_layers_old)))
                    boundary_points_mesh_2 = np.concatenate((boundary_points_mesh_2, np.array(boundary_layers_old)))

                if 'boundary_layer_vertices' in mesh_2.array_names:
                    print('3')
                    boundary_layers_old = mesh_2.points[np.nonzero(mesh_2['boundary_layer_vertices'] == 1)]
                    boundary_points_mesh_2 = np.concatenate((boundary_points_mesh_2, np.array(boundary_layers_old)))
                    boundary_points_mesh_1 = np.concatenate((boundary_points_mesh_1, np.array(boundary_layers_old)))

                vertex_set_mesh_1 = set()
                vertex_set_mesh_2 = set()

                for point in boundary_points_mesh_1:
                    vertex_set_mesh_1.add(tuple(point))
                for point in boundary_points_mesh_2:
                    vertex_set_mesh_2.add(tuple(point))
                vertex_intersection = vertex_set_mesh_1.intersection(vertex_set_mesh_2)

                vertex_dict = {}
                boundary_vertices = np.ones(merged_mesh.n_points, dtype=int)
                boundary_layer_vertices = np.zeros(merged_mesh.n_points, dtype=int)

                for point_idx, point in enumerate(boundary_points):
                    vertex_dict[tuple(point.tolist())] = point_idx
                for vrtx_idx, vrtx in enumerate(merged_mesh.points):
                    if tuple(vrtx.tolist()) not in vertex_dict:
                        boundary_vertices[vrtx_idx] = -1
                        if tuple(vrtx.tolist()) in vertex_intersection:
                            boundary_layer_vertices[vrtx_idx] = 1

                merged_mesh.point_data['boundary_vertices'] = boundary_vertices
                merged_mesh.point_data['boundary_layer_vertices'] = boundary_layer_vertices
            else:

                if 'boundary_layer_vertices' in mesh_1.array_names and 'boundary_layer_vertices' in mesh_2.array_names:
                    merged_mesh = mesh_1.merge(mesh_2, inplace=False)
                elif 'boundary_layer_vertices' in mesh_1.array_names:
                    mesh_2.point_data['boundary_layer_vertices'] = np.zeros(mesh_2.n_points, dtype=int)
                    merged_mesh = mesh_1.merge(mesh_2, inplace=False)
                elif 'boundary_layer_vertices' in mesh_2.array_names:
                    mesh_1.point_data['boundary_layer_vertices'] = np.zeros(mesh_1.n_points, dtype=int)
                    merged_mesh = mesh_2.merge(mesh_1, inplace=False)
                else:
                    merged_mesh = mesh_1.merge(mesh_2, inplace=False)

                idx_mesh = len(self.input_meshes)  # index in self.input_meshes

                if 'boundary_edges_data' in mesh_1.array_names and 'boundary_edges_data' in mesh_2.array_names:
                    edges_1 = mesh_1.extract_all_edges()
                    edges_2 = mesh_2.extract_all_edges()
                    edges_mesh_1 = edges_1.lines.reshape((-1, 3))[:, 1:]
                    edges_mesh_2 = edges_2.lines.reshape((-1, 3))[:, 1:]
                    boundary_edges_mesh_1 = edges_mesh_1[np.nonzero(mesh_1['boundary_edges_data'] == 1)]
                    boundary_edges_mesh_2 = edges_mesh_2[np.nonzero(mesh_2['boundary_edges_data'] == 1)]
                    print(np.max(boundary_edges_mesh_1))
                    print(np.max(boundary_edges_mesh_2))
                    boundary_points_mesh_1 = edges_1.points.reshape((-1, 3))
                    print(len(boundary_points_mesh_1))

                    boundary_points_mesh_2 = edges_2.points.reshape((-1, 3))
                    print(len(boundary_points_mesh_2))

                else:
                    # extract boundary edges and points of mesh_1 and mesh_2
                    boundary_mesh_1 = mesh_1.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                   manifold_edges=False, feature_edges=False)

                    boundary_edges_mesh_1 = boundary_mesh_1.lines.reshape((-1, 3))[:, 1:]

                    boundary_mesh_2 = mesh_2.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                   manifold_edges=False, feature_edges=False)

                    boundary_edges_mesh_2 = boundary_mesh_2.lines.reshape((-1, 3))[:, 1:]

                    boundary_points_mesh_1 = boundary_mesh_1.points
                    boundary_points_mesh_2 = boundary_mesh_2.points

                # extract all edges of merged mesh and concatenate boundary edges and points of mesh_1 and mesh_2
                edges = merged_mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

                boundary_edges_conc = [[boundary_edges_mesh_1] + [boundary_edges_mesh_2]]

                boundary_points_conc = [boundary_points_mesh_1] + [boundary_points_mesh_2]

                edges_dict = [{}, {}]

                # loop over the boundary edges of mesh_1 and mesh_2, mask the edges with the vertex position
                # and create a dictionary with vertex position per edge
                for mesh_idx, boundary_edges in enumerate(boundary_edges_conc[0]):
                    print(boundary_edges)
                    for edge in boundary_edges:
                        print(mesh_idx)
                        print(edge)

                        edge_masked = boundary_points_conc[mesh_idx][edge]
                        print(edge_masked)
                        if tuple(edge_masked[0].tolist()) not in edges_dict[mesh_idx]:
                            edges_dict[mesh_idx][tuple(edge_masked[0].tolist())] = {tuple(edge_masked[1].tolist())}
                        else:
                            edges_dict[mesh_idx][tuple(edge_masked[0].tolist())].update(
                                [tuple(edge_masked[1].tolist())])
                        if tuple(edge_masked[1].tolist()) not in edges_dict[mesh_idx]:
                            edges_dict[mesh_idx][tuple(edge_masked[1].tolist())] = {tuple(edge_masked[0].tolist())}
                        else:
                            edges_dict[mesh_idx][tuple(edge_masked[1].tolist())].update(
                                [tuple(edge_masked[0].tolist())])
                print(edges_dict)
                # loop over edges of the merged mesh and find edges where both meshes share a boundary edge
                boundary_edges_data = np.zeros(len(edges))
                for edge_idx, edge in enumerate(edges):
                    edge_masked = merged_mesh.points[edge]
                    if tuple(edge_masked[0].tolist()) in edges_dict[0] and tuple(edge_masked[0].tolist()) in edges_dict[
                        1]:
                        if tuple(edge_masked[1].tolist()) in edges_dict[0][tuple(edge_masked[0].tolist())] and \
                                tuple(edge_masked[1].tolist()) in edges_dict[1][tuple(edge_masked[0].tolist())]:
                            boundary_edges_data[edge_idx] = 1

                merged_mesh.field_data['boundary_edges_data'] = boundary_edges_data

            self.add_meshes_to_input_and_plotter(merged_mesh, idx_mesh, merged_mesh=True)
        else:
            self.merge_meshes(True)

    def delete_faces(self, check):
        if check:
            self.plotter.enable_cell_picking(callback=self.faces_picker, through=False)
        else:
            self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, show_point=False)

    def faces_picker(self, passed_triangle):
        print(passed_triangle)
        for idx_mesh in self.selected_meshes:
            vertex_dict = {}
            faces_dict = {}
            points = self.input_meshes[idx_mesh].points
            faces = self.input_meshes[idx_mesh].faces.reshape((-1, 4))[:, 1:4]
            for point_idx, point in enumerate(points):
                vertex_dict[tuple(point.tolist())] = point_idx
            for face_idx, face in enumerate(faces):
                faces_dict[tuple(sorted(face.tolist()))] = face_idx
            face_idx = self.find_face_index(passed_triangle, vertex_dict, faces_dict)
            if face_idx is not None:
                self.input_meshes[idx_mesh].remove_cells(face_idx, inplace=True)
                self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                      name=f"mesh_{idx_mesh}", opacity=100)
                break

    def mesh_decimation(self, check):
        if check:
            button_idx = self.button_widgets_utils['mesh_decimation']
            if self._state != States.picking_selection and self._state != States.deleting_selection:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                return
            if len(self.selected_meshes) < 1:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                print('Please select a mesh')
                return
            elif len(self.selected_meshes) > 1:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                print('Please select only one mesh')
                return

            self.clear_selection(True)
            self.set_state(States.mesh_decimation)
            temp_mesh = self.input_meshes[self.selected_meshes[0]]
            self.deselect_all_meshes(True)
            pv.save_meshio("temp_mesh.obj", temp_mesh)

            # convert to meshlab
            # define a helper function for the button to call the set_percent_decimation function
            def set_reduction_helper(check_1):
                if check_1:
                    msg = 'Set reduction factor: (float)'
                    # user_input = float(input("Set reduction factor: ")) or 0.1
                    successful_input = False
                    while not successful_input:
                        user_input = easygui.enterbox(msg, default='0.1')
                        if user_input is None:
                            return
                        else:
                            try:
                                user_input = float(user_input)
                                successful_input = True
                            except ValueError:
                                print("Input is not a float, please try again or cancel")

                    idx_slider = self.button_widgets_utils['mesh_decimation_slider']
                    self.plotter.slider_widgets[idx_slider].GetRepresentation().SetValue(user_input)
                    self.set_percent_decimation(user_input)
                else:
                    set_reduction_helper(True)

            if 'mesh_decimation_slider' not in self.button_widgets_utils:
                # if called first time initialize the slider and buttons
                self.plotter.add_slider_widget(self.set_percent_decimation, rng=(0.0, 1.0))
                self.button_widgets_utils['mesh_decimation_slider'] = len(self.plotter.slider_widgets) - 1

                self.plotter.add_checkbox_button_widget(set_reduction_helper, position=(800, 10), color_on='orange',
                                                        color_off='grey')
                self.plotter.add_text('set reduction ', font_size=6, position=(780, 60), name='set_reduction_button')
                self.button_widgets_utils['mesh_decimation_reduction'] = len(self.plotter.button_widgets) - 1

                self.plotter.add_checkbox_button_widget(self.abort_decimation, position=(900, 10), color_on='orange',
                                                        color_off='orange')
                self.plotter.add_text('abort decimation', font_size=6, position=(880, 60),
                                      name='abort_decimation_button')
                self.button_widgets_utils['mesh_abort_decimation'] = len(self.plotter.button_widgets) - 1

            else:

                # activate slider and button

                slider_idx = self.button_widgets_utils['mesh_decimation_slider']
                self.plotter.slider_widgets[slider_idx].GetRepresentation().SetVisibility(True)
                self.plotter.slider_widgets[slider_idx].GetRepresentation().SetValue(0.5)
                self.plotter.slider_widgets[slider_idx].On()

                self.plotter.add_text('set reduction ', font_size=6, position=(780, 60), name='set_reduction_button')
                button_idx = self.button_widgets_utils['mesh_decimation_reduction']

                self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOn()
                self.plotter.button_widgets[button_idx].On()

                self.plotter.add_text('abort decimation', font_size=6, position=(880, 60),
                                      name='abort_decimation_button')
                button_idx = self.button_widgets_utils['mesh_abort_decimation']
                self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOn()
                self.plotter.button_widgets[button_idx].On()
                self.set_percent_decimation(0.5)

            # Button for setting reduction

        else:
            try:
                # testing if it is possible to subdivide the mesh -> decimation may produce topology which causes
                # problems during subdivision or if the mesh is to large memory allocation error can occur
                mesh_for_subdivision = files.read("temp_mesh_2.obj")
                mesh_for_subdivision.subdivide()
            except MemoryError as m:
                raise MemoryError(m, 'Subdivision might fail on the mesh, try to decrease the number of vertices')
            except IndexError as i:
                raise IndexError(i, 'Decimation might have created bad topology, please try to increase the reduction'
                                    'factor when possible.')
            else:
                # add the mesh to the input meshes and change the state
                # if None in self.input_meshes:
                #     idx_mesh = self.input_meshes.index(None)
                # else:

                idx_mesh = len(self.input_meshes)

                # self.new_meshes_idx[idx_mesh] = len(self.new_meshes_idx.keys())
                self.add_meshes_to_input_and_plotter(self.decimated_mesh, idx_mesh, new_mesh=True)

                # deactivate slider, buttons and delete temp. meshes
                self.abort_decimation(True)

    def set_percent_decimation(self, reduction_percent=0.5):
        # initilize pymeshlab set for performing decimation
        ms = pymeshlab.MeshSet()
        # load the tempory file
        ms.load_new_mesh("temp_mesh.obj")
        # decimation with the passed reduction
        ms.simplification_quadric_edge_collapse_decimation(targetperc=reduction_percent)
        # save the decimated mesh temporary
        ms.save_current_mesh("temp_mesh_2.obj")
        # read again with pyvista
        self.decimated_mesh = pv.read("temp_mesh_2.obj")
        # plot the mesh
        self.plotter.add_mesh(self.decimated_mesh, color='orange', name='decimated_mesh', show_edges=True)

    def abort_decimation(self, check):
        if check:
            # set state back to picking
            self.set_state(States.picking_selection)
            # deactivate the slider and button
            slider_idx = self.button_widgets_utils['mesh_decimation_slider']
            self.plotter.slider_widgets[slider_idx].GetRepresentation().SetVisibility(False)
            self.plotter.slider_widgets[slider_idx].Off()

            self.plotter.remove_actor('set_reduction_button')
            button_idx = self.button_widgets_utils['mesh_decimation_reduction']
            self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOff()
            self.plotter.button_widgets[button_idx].Off()

            self.plotter.remove_actor('abort_decimation_button')
            button_idx = self.button_widgets_utils['mesh_abort_decimation']
            self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOff()
            self.plotter.button_widgets[button_idx].Off()

            button_idx = self.button_widgets_utils['mesh_decimation']
            self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)

            # remove additional mesh and delete temporary files
            self.decimated_mesh = pv.PolyData()
            self.plotter.remove_actor('decimated_mesh')
            os.remove("temp_mesh.obj")
            os.remove("temp_mesh_2.obj")
        else:
            self.abort_decimation(True)

    def constrain_selection_min_max(self, check, idx_mesh):
        if check:
            mesh = self.input_meshes[idx_mesh]

            button_boundary_idx = self.button_widgets_boundary[f"button_boundary_{idx_mesh}"]
            self.plotter.button_widgets[button_boundary_idx].GetRepresentation().SetState(False)

            # deactivate mesh for picking
            self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100)

            if 'extrema' in mesh.array_names:
                extrema_points_idx = np.nonzero(mesh["extrema"] == 1)
                if len(extrema_points_idx) == 0:
                    # no min max found; make whole mesh pickable again
                    self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                          name=f"mesh_{idx_mesh}", opacity=100)
                else:
                    extrema_points = mesh.points[extrema_points_idx]
                    self.plotter.add_points(extrema_points, pickable=True, name=f"constrained_mesh{idx_mesh}")
            else:
                extrema = np.zeros(mesh.n_points)
                points = mesh.points
                edges = mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]
                edges_extracted = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                             feature_edges=False,
                                                             manifold_edges=False)
                boundary_vertices_set = set()
                boundary_vertices = edges_extracted.points.tolist()
                for vertex in boundary_vertices:
                    boundary_vertices_set.add(tuple(vertex))
                vertex_vertex_dictionary = data_structure.vertex_vertex_dictionary(points, edges)
                local_minima = []
                local_maxima = []

                for i, vertex in enumerate(points):
                    if not tuple(vertex) in boundary_vertices_set:
                        is_extrema = []
                        vertex_z = vertex[2]
                        for vertex_idx in vertex_vertex_dictionary[i]:
                            neighbor_z = points[vertex_idx][2]
                            if neighbor_z > vertex_z:
                                is_extrema.append('Min')
                            elif neighbor_z < vertex_z:
                                is_extrema.append('Max')
                        is_extrema = np.array(is_extrema)
                        if all(is_extrema == 'Min'):
                            local_minima.append(i)
                        elif all(is_extrema == 'Max'):
                            local_maxima.append(i)

                    else:
                        is_extrema = []
                        vertex_z = vertex[2]
                        for vertex_idx in vertex_vertex_dictionary[i]:
                            if tuple(points[vertex_idx]) in boundary_vertices_set:
                                neighbor_z = points[vertex_idx][2]
                                if neighbor_z > vertex_z:
                                    is_extrema.append('Min')
                                elif neighbor_z < vertex_z:
                                    is_extrema.append('Max')
                        is_extrema = np.array(is_extrema)
                        if len(is_extrema) > 0:
                            if all(is_extrema == 'Min'):
                                local_minima.append(i)
                            elif all(is_extrema == 'Max'):
                                local_maxima.append(i)

                local_extrema = local_minima + local_maxima
                extrema[local_extrema] = 1
                mesh.point_data['extrema'] = extrema
                if len(local_extrema) == 0:
                    # no min max found; make whole mesh pickable again
                    self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                          name=f"mesh_{idx_mesh}", opacity=100)
                else:
                    extrema_points = points[local_extrema]
                    self.plotter.add_points(extrema_points, pickable=True, name=f"constrained_mesh{idx_mesh}")
        else:
            # remove actor for vertices on min max and make the mesh pickable again
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100)
            self.plotter.remove_actor(f"constrained_mesh{idx_mesh}")

    def delete_mesh(self, check, idx_mesh):
        if idx_mesh in self.invisible_meshes:
            print(f"mesh_{idx_mesh} is invisible, please make visible again to delete")
            button_idx = self.button_widgets[f"button_deletion{idx_mesh}"]
            self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
            return
        if check:
            if idx_mesh in self.selected_meshes:
                self.select_mesh(False, idx_mesh)
            self.input_meshes[idx_mesh] = None
            if idx_mesh in self.new_meshes_idx:
                self.new_meshes_idx[idx_mesh] = None
            if idx_mesh in self.merged_meshes_idx:
                self.merged_meshes_idx[idx_mesh] = None

            self.plotter.remove_actor(f"mesh_name_{idx_mesh}")
            self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-points')
            self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-labels')
            self.plotter.button_widgets[self.button_widgets[f"button_{idx_mesh}"]] = None
            self.plotter.button_widgets[self.button_widgets[f"button_invisible_{idx_mesh}"]] = None
            self.plotter.button_widgets[self.button_widgets[f"button_deletion{idx_mesh}"]] = None
            self.plotter.button_widgets[self.button_widgets_boundary[f"button_boundary_{idx_mesh}"]] = None
            self.plotter.remove_actor(f"mesh_{idx_mesh}")
        else:
            self.delete_mesh(True)

    def save_mesh(self, check):
        if check:

            title = 'Save a mesh'
            msg = 'select a mesh for saving'
            choices = [f"mesh_{i}" for i, mesh in enumerate(self.input_meshes) if mesh is not None]
            if len(choices) == 0:
                return
            elif len(choices) == 1:
                choice = choices[0]
            else:
                choice = easygui.choicebox(msg, title, choices)
            if choice is None:
                return
            mesh_idx = int(choice.split('_')[1])

            msg = 'save mesh as .obj file'
            filename = easygui.filesavebox(msg, default=f"{choice}.obj", filetypes="\\*.obj")
            if filename is None:
                return
            if not filename.endswith('.obj'):
                filename += ".obj"

            pv.save_meshio(filename, self.input_meshes[mesh_idx])
            temp_pysubdiv = files.read(filename)
            try:
                temp_pysubdiv.data['dynamic_vertices'] = self.input_meshes[mesh_idx]['dynamic_vertices']
                temp_pysubdiv.data['fitting_method'] = 'dynamic_faces'
            except KeyError as k:
                print(k)
            try:
                temp_pysubdiv.data['dynamic_faces'] = self.input_meshes[mesh_idx]['dynamic_faces']
            except KeyError as k:
                print(k)
            try:
                temp_pysubdiv.data['boundary_vertices'] = self.input_meshes[mesh_idx]['boundary_vertices']
            except KeyError as k:
                print(k)
            try:
                temp_pysubdiv.data['non_manifold_vertices'] = self.input_meshes[mesh_idx]['non_manifold_vertices']
            except KeyError as k:
                print(k)
            try:
                temp_pysubdiv.data['extrema'] = self.input_meshes[mesh_idx]['extrema']
            except KeyError as k:
                print(k)
            try:
                temp_pysubdiv.data['boundary_layer_vertices'] = self.input_meshes[mesh_idx]['boundary_layer_vertices']
            except KeyError as k:
                print(k)
            try:
                temp_pysubdiv.data['boundary_edges_data'] = self.input_meshes[mesh_idx]['boundary_edges_data']
            except KeyError as k:
                print(k)

            temp_pysubdiv.save_data(filename[:-4] + "Data")
        else:
            self.save_mesh(True)

    def load_mesh(self, check):
        if check:
            msg = 'Select a .obj File for loading'
            file = easygui.fileopenbox(msg, filetypes="*.obj")
            if file is None:
                return
            elif not file.endswith('.obj'):
                print('The file is not in .obj format, please try again')
            else:
                mesh = pv.read(file)

                file_path = file.rsplit("/", 1)
                path_txt = [file for file in sorted(os.listdir(file_path[0])) if file.endswith(".pkl")]

                idx_mesh = len(self.input_meshes)
                idx_mesh = self.add_meshes_to_input_and_plotter(mesh, idx_mesh)

                msg = 'Load additional data for the file?'
                user_bool = easygui.ynbox(msg)

                if user_bool:

                    if file_path[1][:-4] + "Data.pkl" in path_txt:
                        file_data = file_path[0] + "/" + file_path[1][:-4] + "Data.pkl"
                    else:
                        msg = 'Select data file (.pkl) for the mesh'
                        file_data = easygui.fileopenbox(msg, filetypes="*.pkl")
                    if file_data is None:
                        pass
                    else:
                        while not file_data.endswith('pkl'):
                            msg = 'The file is not in .pkl format, do you want to try again?'
                            user_bool = easygui.ynbox(msg)
                            if not user_bool:
                                break
                            else:
                                msg = 'Select data file (.pkl) for the mesh'
                                file_data = easygui.fileopenbox(msg, filetypes="*.pkl")
                        temp_mesh = files.read(file)
                        temp_mesh.load_data(file_data)
                        if 'dynamic_vertices' in temp_mesh.data:
                            self.input_meshes[idx_mesh].point_data['dynamic_vertices'] = temp_mesh.data[
                                'dynamic_vertices']
                        if 'dynamic_faces' in temp_mesh.data:
                            self.input_meshes[idx_mesh].cell_data['dynamic_faces'] = temp_mesh.data['dynamic_faces']
                        if 'boundary_vertices' in temp_mesh.data:
                            self.input_meshes[idx_mesh].point_data['boundary_vertices'] = temp_mesh.data[
                                'boundary_vertices']
                        if 'boundary_layer_vertices' in temp_mesh.data:
                            self.input_meshes[idx_mesh].point_data['boundary_layer_vertices'] = temp_mesh.data[
                                'boundary_layer_vertices']

                        if 'boundary_edges_data' in temp_mesh.data:
                            self.input_meshes[idx_mesh].field_data['boundary_edges_data'] = temp_mesh.data[
                                'boundary_edges_data']

                        # if 'non_manifold_vertices' in temp_mesh.data:
                        #     self.input_meshes[idx_mesh].cell_data['non_manifold_vertices'] = temp_mesh.data[
                        #         'non_manifold_vertices']
                        if 'extrema' in temp_mesh.data:
                            self.input_meshes[idx_mesh].point_data['extrema'] = temp_mesh.data['extrema']
                else:
                    pass

    def main(self):
        offset = 0
        for idx, mesh in enumerate(self.input_meshes):
            self.plotter.add_mesh(mesh, color='green', pickable=False, show_edges=True, name=f"mesh_{idx}")
            self.plotter.add_point_labels([mesh.center_of_mass()], [f"mesh_{idx}"], name=f"labels_mesh_{idx}")
            self.plotter.add_text(f"mesh_{idx}", position=(10, 700 - offset), font_size=10, name=f"mesh_name_{idx}")

            vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.mesh_visibility, idx_mesh=idx,
                                                     position=(10, 680 - offset), size=15, border_size=2)

            self.button_widgets[f"button_invisible_{idx}"] = len(self.plotter.button_widgets) - 1

            vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.select_mesh, idx_mesh=idx,
                                                     position=(30, 680 - offset), size=15, border_size=2,
                                                     color_on='red')

            self.button_widgets[f"button_{idx}"] = len(self.plotter.button_widgets) - 1

            vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.constrain_selection_to_boundary,
                                                     idx_mesh=idx, position=(50, 680 - (idx * 50)), size=15,
                                                     border_size=2)

            # save the index of the button in the self.plotter.button_widgets property for deletion
            self.button_widgets_boundary[f"button_boundary_{idx}"] = len(self.plotter.button_widgets) - 1

            button_boundary_idx = self.button_widgets_boundary[f"button_boundary_{idx}"]
            # set state and turn visibility of button off
            self.plotter.button_widgets[button_boundary_idx].Off()
            self.plotter.button_widgets[button_boundary_idx].GetRepresentation().SetState(False)
            self.plotter.button_widgets[button_boundary_idx].GetRepresentation().VisibilityOff()

            vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.constrain_selection_min_max,
                                                     idx_mesh=idx, position=(70, 680 - (idx * 50)), size=15,
                                                     border_size=2)

            # save the index of the button in the self.plotter.button_widgets property for deletion
            self.button_widgets[f"button_min_max_{idx}"] = len(self.plotter.button_widgets) - 1

            button_min_max_idx = self.button_widgets[f"button_min_max_{idx}"]
            # set state and turn visibility of button off
            self.plotter.button_widgets[button_min_max_idx].Off()
            self.plotter.button_widgets[button_min_max_idx].GetRepresentation().SetState(False)
            self.plotter.button_widgets[button_min_max_idx].GetRepresentation().VisibilityOff()

            vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.delete_mesh, idx_mesh=idx,
                                                     position=(90, 680 - offset), size=15, border_size=2,
                                                     color_on='red')

            self.button_widgets[f"button_deletion{idx}"] = len(self.plotter.button_widgets) - 1

            offset += 50

        self.plotter.add_checkbox_button_widget(self.triangulate_selection, color_on='purple', color_off='purple')
        self.plotter.add_text('triangulation', font_size=6, position=(6, 60), name="triangulate_text")
        self.button_widgets_utils['triangulation_button'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.selection_undo, position=(150, 10))
        self.button_widgets_utils['polygon_selection_undo'] = len(self.plotter.button_widgets) - 1

        # button switch between selection and deletion
        self.plotter.add_checkbox_button_widget(self.switch_picking_deleting, position=(200, 10), value=True)
        self.plotter.add_text('select', font_size=6, position=(210, 60), name="selection_text")
        self.button_widgets_utils['button_picking_deleting'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.deselect_all_meshes, position=(300, 10))
        self.button_widgets_utils['deselect_all_meshes'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.clear_selection, position=(400, 10))
        self.button_widgets_utils['clear_selection'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.merge_meshes, position=(500, 10), color_on='yellow',
                                                color_off='yellow')
        self.button_widgets_utils['merge_meshes'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.delete_faces, position=(600, 10), color_on='purple',
                                                color_off='grey')
        self.plotter.add_text('delete faces', font_size=6, position=(600, 60), name="delete_faces_text")
        self.button_widgets_utils['delete_faces'] = len(self.plotter.button_widgets) - 1

        # Buttons + slider for mesh decimation
        self.plotter.add_checkbox_button_widget(self.mesh_decimation, position=(700, 10), color_on='orange',
                                                color_off='grey')
        self.plotter.add_text('mesh decimation', font_size=6, position=(680, 60))
        self.button_widgets_utils['mesh_decimation'] = len(self.plotter.button_widgets) - 1

        # Button for saving a cotrol_mesh
        self.plotter.add_checkbox_button_widget(self.save_mesh, position=(1000, 10), color_on='lightblue',
                                                color_off='lightblue')
        self.plotter.add_text('save mesh', font_size=6, position=(990, 60))
        self.button_widgets_utils['save_mesh'] = len(self.plotter.button_widgets) - 1
        # Button for loading a mesh
        self.plotter.add_checkbox_button_widget(self.load_mesh, position=(1100, 10), color_on='green',
                                                color_off='green')
        self.plotter.add_text('load mesh', font_size=6, position=(1090, 60))
        self.button_widgets_utils['load_mesh'] = len(self.plotter.button_widgets) - 1

        # # Button for showing curvature
        # self.plotter.add_checkbox_button_widget(self.show_curvature, position=(1200, 10), color_on='green',
        #                                         color_off='green')
        # self.plotter.add_text('show curvature', font_size=6, position=(1190, 60))
        # self.button_widgets_utils['show curvature'] = len(self.plotter.button_widgets) - 1

        self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, pickable_window=False,
                                          show_point=False)
        self.plotter.add_camera_orientation_widget()
        self.set_state(States.picking_selection)
        self.plotter.isometric_view_interactive()
        self.plotter.show()

    def set_state(self, state):
        self._state = state

    @staticmethod
    def find_face_index(picked_face_passed, vertex_dict, faces_dict):
        picked_face_vertices = picked_face_passed.points
        vertex_indices_from_mesh = []
        for vert in picked_face_vertices:
            if tuple(vert.tolist()) not in vertex_dict:
                return
            vertex_indices_from_mesh.append(vertex_dict[tuple(vert.tolist())])
        face_idx = faces_dict[tuple(sorted(vertex_indices_from_mesh))]
        return face_idx


def define_volumes(mesh):
    """
    Define volumes for gmesh software. Volumes are set by faces which form a manifold part of the mesh. Faces which form
    a volume can be picked in the viewer by hovering over with the mouse and pressing  the 'P' key.
    After picking the user is asked to provide the index/number of volume which the picked face should be assigned to by
    console input. An integer should be provided.

    The function will return a nested list with n-number of volumes. Whereas the index of the list is the index/number
    of the volume and the elements in the nested list are the indices of the faces forming the volume.
    Important: If a volume index/number one must remember that the returned list is continuous. E.g. if volumes 0, 1, 3
    are defined the returned list will have the indices 0, 1, 2, aso volume 3 becomes index 2 in the list.

    Parameters
    --------------
    mesh : PySubdiv mesh object
        mesh on which the volumes are defined

    Returns
    --------------
    volumes_list: nested list of n volumes and the indices of the faces forming the volume

    """
    edges_face_incidence = mesh.data['faces_edges_incidence_list']
    faces_edge_incidence = mesh.data['faces_incidence_list']
    faces_faces_incidence = []
    # model = mesh.model()  # converting PySubdiv mesh to PyVista Polydata
    triangles = []  # list to hold triangles of the mesh
    invisible_triangles = []  # list to hold deleted triangles
    model_vertices = mesh.vertices[mesh.faces]  # vertices of the mesh indexed by faces
    triangles_colors = {}  # dictionary to hold the color of one triangle
    colors = ['blue', 'yellow', 'red', 'orange', 'purple', 'cyan', 'pink', 'beige']

    for i in range(len(model_vertices)):  # define each triangle and append
        triangles_colors[i] = 'green'
        triangles.append(pv.PolyData(model_vertices[i], [3, 0, 1, 2]))
        edges_around_face = edges_face_incidence[i]
        faces_around = set()
        for edge in edges_around_face:
            for face in faces_edge_incidence[edge]:
                if face != i:
                    faces_around.add(face)
        faces_faces_incidence.append(faces_around)

    vertex_dict = {}
    faces_dict = {}
    volumes = {}  # volumes for gmesh
    volume_index_used = []  # list for storing the volumes_idx used for undoing the last action
    selected_cells = []  # list to store selection

    index = 0
    for vertex in mesh.vertices:
        vertex_dict[tuple(vertex.tolist())] = index
        index += 1

    index = 0
    for face in mesh.faces:
        faces_dict[tuple(sorted(face.tolist()))] = index
        index += 1

    # callback function to compare the picked face against the faces of the mesh to find the correct index of the face
    # of the mesh

    def make_selection(picked_face):
        if isinstance(picked_face, type(None)):
            return
        else:
            if isinstance(picked_face, pyvista.core.composite.MultiBlock):
                for cell in picked_face:
                    face_idx = find_face_index(cell)
                    if face_idx in selected_cells:
                        selected_cells.remove(face_idx)
                        p.remove_actor(f"tris_selection_{face_idx}")
                    else:
                        p.add_mesh(triangles[face_idx].extract_feature_edges(), color='white', edge_color='white',
                                   line_width=7, pickable=False, name=f"tris_selection_{face_idx}", scalars=None)
                        selected_cells.append(face_idx)
            else:
                face_idx = find_face_index(picked_face)
                if face_idx in selected_cells:
                    selected_cells.remove(face_idx)
                    p.remove_actor(f"tris_selection_{face_idx}")
                else:
                    p.add_mesh(triangles[face_idx].extract_feature_edges(), color='white', edge_color='white',
                               line_width=7, pickable=False, name=f"tris_selection_{face_idx}", scalars=None)
                    selected_cells.append(face_idx)

    def automatic_selection(check):
        if check:
            if len(selected_cells) == 1:
                start_face = selected_cells[0]
                visited_faces = []
                faces_to_visit = {start_face}
                counter = 0
                connected_faces = np.array(list(faces_faces_incidence[start_face]))
                next_face = start_face

                while len(faces_to_visit) > 0:
                    if counter == 0:
                        faces_to_visit = set()

                    continue_with_next_face = False

                    if len(connected_faces) > 3:
                        connected_faces = []
                        connected_edges = edges_face_incidence[next_face]
                        for connected_edge in connected_edges:
                            if len(faces_edge_incidence[connected_edge]) > 2:
                                pass
                            else:
                                for face_index in faces_edge_incidence[connected_edge]:
                                    if face_index != next_face and face_index not in visited_faces:
                                        connected_faces.append(face_index)

                        connected_faces = np.unique(connected_faces)
                    if len(connected_faces) == 0:
                        next_face = list(faces_to_visit)[np.random.randint(len(faces_to_visit))]
                        faces_to_visit.remove(next_face)
                        rest_of_faces = []
                        connected_edges = edges_face_incidence[next_face]
                        for connected_edge in connected_edges:
                            if len(faces_edge_incidence[connected_edge]) > 2:
                                pass
                            else:
                                for face_index in faces_edge_incidence[connected_edge]:
                                    if face_index != next_face and face_index not in visited_faces:
                                        rest_of_faces.append(face_index)
                        continue_with_next_face = True

                    else:
                        next_face = connected_faces[np.random.randint(len(connected_faces))]
                        rest_of_faces = connected_faces[np.nonzero(connected_faces != next_face)[0]]  # 2

                    while not continue_with_next_face:
                        if len(rest_of_faces) == 0:
                            if len(faces_to_visit) == 0:
                                print('Oops, no faces to visit anymore. This can happen when a non-manifold face is \n'
                                      'selected first. Please try again with a manifold face')
                                return
                            next_face = list(faces_to_visit)[np.random.randint(len(faces_to_visit))]
                            rest_of_faces = []
                            connected_edges = edges_face_incidence[next_face]
                            for connected_edge in connected_edges:
                                if len(faces_edge_incidence[connected_edge]) > 2:
                                    pass
                                else:
                                    for face_index in faces_edge_incidence[connected_edge]:
                                        if face_index != next_face and face_index not in visited_faces:
                                            rest_of_faces.append(face_index)
                            faces_to_visit.remove(next_face)
                            continue_with_next_face = True
                        else:
                            if next_face in visited_faces:
                                if isinstance(rest_of_faces, np.int64):
                                    next_face = rest_of_faces
                                else:
                                    next_face = rest_of_faces[np.random.randint(len(rest_of_faces))]  # 1
                                    temp_list = []
                                    for i in rest_of_faces:
                                        if i != next_face and i not in visited_faces:
                                            temp_list.append(i)
                                    rest_of_faces = np.array(temp_list)

                            else:
                                continue_with_next_face = True

                    if isinstance(rest_of_faces, np.int64):
                        if rest_of_faces not in visited_faces:
                            faces_to_visit.add(rest_of_faces)
                    else:
                        for rest_face in rest_of_faces:
                            if rest_face not in visited_faces:
                                faces_to_visit.add(rest_face)

                    visited_faces.append(next_face)
                    if next_face not in selected_cells:
                        p.add_mesh(triangles[next_face].extract_feature_edges(), color='white', edge_color='white',
                                   line_width=7, pickable=False, name=f"tris_selection_{next_face}", scalars=None)
                        selected_cells.append(next_face)
                    connected_faces = np.array(list(faces_faces_incidence[next_face]))
                    counter += 1

            else:
                print('No cell or more than one face selected, please select only one face')
        else:
            automatic_selection(True)

    def clear_selection(check):
        if check:
            for selection in selected_cells:
                p.remove_actor('tris_selection_' + str(selection))
                # p.add_mesh(triangles[selection], color='green', use_transparency=False, show_edges=True,
                # name="tris_" + str(selection))
            selected_cells.clear()
        else:
            clear_selection(True)

    def find_face_index(picked_face_passed):
        picked_face_vertices = picked_face_passed.points
        vertex_indices_from_mesh = []
        for vert in picked_face_vertices:
            vertex_indices_from_mesh.append(vertex_dict[tuple(vert.tolist())])
        face_idx = faces_dict[tuple(sorted(vertex_indices_from_mesh))]
        return face_idx

    def define_volume(face_idx, volume_index):
        if volume_index in volumes:
            if face_idx in volumes[volume_index]:
                print(f"face {face_idx} already part of volume {volume_index}")
            else:
                volumes[volume_index].append(face_idx)
                print(f'face {face_idx} added to volume {volume_index}')
                print(f'volume {volume_index} contain following faces: {volumes[volume_index]}')
        else:
            volumes[volume_index] = [face_idx]
            print(f'face {face_idx} added to volume {volume_index}')
            print(f'volume {volume_index} contain following faces: {volumes[volume_index]}')

        volume_index_used.append(volume_index)
        triangles_colors[face_idx] = colors[volume_index]
        p.add_mesh(triangles[face_idx], use_transparency=False, show_edges=True,
                   name="tris_" + str(face_idx), color=colors[volume_index])

    def undo_last_face(check):  # callback function to undo last face
        if check:
            if not volume_index_used:  # check if list is not empty
                print('No volumes defined')
            else:
                print(volume_index_used)
                print(volumes)
                volume_idx = volume_index_used.pop()  # pop last volume number of the list and store it
                face_idx, volumes[volume_idx] = volumes[volume_idx][-1], volumes[volume_idx][
                                                                         :-1]  # delete last added face
                p.add_mesh(triangles[face_idx], use_transparency=False, show_edges=True, name="tris_" + str(face_idx),
                           color='green')
                print('face ', face_idx, 'deleted from volume ', volume_idx)
                print('volume', volume_idx, 'contains faces', volumes[volume_idx])
        else:
            undo_last_face(True)

    def print_volumes(check):  # callback function to print out the volume dictionary
        if check:
            print('volumes with contained faces \n', volumes)
        else:
            print_volumes(True)

    def delete_face(face_idx):
        invisible_triangles.append(face_idx)
        p.add_mesh(triangles[face_idx], color=triangles_colors[face_idx], use_transparency=True, opacity=0.8,
                   show_edges=True,
                   pickable=False, name="tris_" + str(face_idx))

    def activate_volumes(check):
        if check:
            if len(selected_cells) == 0:
                print('No faces selected')
            else:
                while True:
                    volume_index = input("volume number (integer) or 'c' to cancel: ") or 'c'
                    try:
                        if volume_index[0].lower() == 'c':
                            return
                        else:
                            volume_index = int(volume_index)
                            break
                    except ValueError:
                        print(
                            f'The input {volume_index} cannot be interpreted as an integer, please try again or enter '
                            f'"cancel" to cancel')
                for cell in selected_cells:
                    define_volume(cell, volume_index)
        else:
            activate_volumes(True)

    def make_triangles_visible(check):
        if check:
            if len(invisible_triangles) == 0:
                print('All triangles are visible')
            else:
                for idx_invisible_tris in invisible_triangles:
                    p.add_mesh(triangles[idx_invisible_tris], color=triangles_colors[idx_invisible_tris],
                               use_transparency=False, show_edges=True, name="tris_" + str(idx_invisible_tris))
                invisible_triangles.clear()
        else:
            make_triangles_visible(True)

    def make_invisible(check):
        if check:
            for cell in selected_cells:
                delete_face(cell)
        else:
            make_invisible(True)

    p = pv.Plotter()
    p.set_background('royalblue', top='aliceblue')

    for i in range(len(triangles)):
        p.add_mesh(triangles[i], color='green', use_transparency=False, show_edges=True, name="tris_" + str(i))

    p.enable_cell_picking(callback=make_selection, through=False, show_message=False)
    p.add_text('Press P to select face under the mouse courser.\n'
               'Press R to toggle to rectangle based selection.', color='black')

    p.add_checkbox_button_widget(activate_volumes, value=True, position=(10, 250), size=35,
                                 color_off='red', color_on='red')
    p.add_text('set volumes', position=(60, 253), font_size=14, color='black')

    p.add_checkbox_button_widget(make_invisible, value=True, position=(10, 210), size=35,
                                 color_off='blue', color_on='blue')
    p.add_text('make selection invisible', position=(60, 213), font_size=14, color='black')

    p.add_checkbox_button_widget(clear_selection, position=(10, 170), color_off='green',
                                 color_on='green', size=35)
    p.add_text('clear selection', position=(60, 173), font_size=14, color='black')

    p.add_checkbox_button_widget(automatic_selection, position=(10, 130), color_off='purple',
                                 color_on='purple', size=35)

    p.add_text('automatic selection', position=(60, 133), font_size=14, color='black')

    p.add_checkbox_button_widget(make_triangles_visible, position=(10, 90), color_on='orange',
                                 color_off='orange', size=35)

    p.add_text('make triangles visible', position=(60, 93), font_size=14, color='black')

    p.add_checkbox_button_widget(undo_last_face, position=(10, 50), color_on='lightblue',
                                 color_off='lightblue', size=35)
    p.add_text('undo last face', position=(60, 53), font_size=14, color='black')

    p.add_checkbox_button_widget(print_volumes, position=(10.0, 10.0), color_on='brown', color_off='brown', size=35)
    p.add_text('print volumes', position=(60, 13), font_size=14, color='black')

    p.show()
    sorted_volumes = dict(sorted(volumes.items()))  # sort the dict of volumes
    volumes_list = list(sorted_volumes.values())  # convert dictionary to list, skipped volume number is reduces
    return volumes_list


class MyMainWindow(MainWindow):

    def __init__(self, input_meshes, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.showMaximized()
        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        self.plotter.background_color = "dimgrey"

        vlayout.addWidget(self.plotter.interactor)

        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)
        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_load_mesh = QtWidgets.QAction('Load Mesh', self)
        self.add_load_mesh.triggered.connect(self.load_mesh)
        meshMenu.addAction(self.add_load_mesh)
        self.add_save_mesh = QtWidgets.QAction('Save Mesh', self)
        self.add_save_mesh.triggered.connect(self.save_mesh)
        meshMenu.addAction(self.add_save_mesh)
        self.visualize_data = QtWidgets.QAction('Visualize data array', self)
        self.visualize_data.triggered.connect(self.visualize_data_array)
        meshMenu.addAction(self.visualize_data)

        self.render_as_sphere = QtWidgets.QAction('Render points of mesh only', self)
        self.render_as_sphere.triggered.connect(self.render_selected_mesh_as_spheres)
        meshMenu.addAction(self.render_as_sphere)

        self.subdivide_mesh = QtWidgets.QAction('Subdivide a mesh', self)
        self.subdivide_mesh.triggered.connect(self.subdivide)
        meshMenu.addAction(self.subdivide_mesh)

        self.delete_faces_action = QtWidgets.QAction('Delete selected faces', self)
        self.delete_faces_action.triggered.connect(self.delete_faces)
        meshMenu.addAction(self.delete_faces_action)

        self.volumes_action = QtWidgets.QAction('Define volumes for mesh', self)
        self.volumes_action.triggered.connect(self.define_volumes)
        meshMenu.addAction(self.volumes_action)

        if show:
            self.show()

        self.input_meshes = self.check_meshes(input_meshes)
        self.input_meshes_data = {}  #
        # self.plotter = pv.Plotter()  # PyVista.Plotter
        self.selection = []  # empty list to store selection of vertices
        self.selection_idx = {}  # empty list to store idx of selected vertices
        self.selected_meshes = []  # list to store selection of meshes
        self.invisible_meshes = []  # keep track of invisible meshes
        self.button_widgets_invisible = []  # button widgets of meshes
        self.button_widgets = {}
        self.button_widgets_boundary = {}  # index of button widget in plotter.button_widgets
        self.button_widgets_utils = {}  # button widget and index in plotter.button_widgets
        # self.selection_polygon = []  # list to store selection for polygon cutting
        self.selection_polygon_idx = []  # list for indices for cutting polygon
        self.selection_polygon_mesh = pv.PolyData()
        self.selection_mesh = pv.PolyData()
        self.new_meshes_idx = {}  # key index new mesh -> value index in self.input_meshes
        self.merged_meshes_idx = {}  # key index merged mesh -> value index in self.input_meshes
        self.dynamic_vertices = {}
        self.decimated_mesh = pv.PolyData()  # field to hold the decimated mesh
        self._state = None
        # dictionaries for storing the different buttons
        self.mesh_text_dic = {}
        self.visibility_button_dict = {}
        self.select_button_dict = {}
        self.select_constrain_button_dict = {}
        self.select_constrain_min_max_dict = {}
        self.delete_button = {}
        self.selected_faces = {}
        self.control_points = None
        self.mover = None
        self.mover_btn = {}  # storing mover button
        self.edges_btn = {}  # storing edges button
        self.edges_mesh = None  # reference for the edge selection
        self.selected_edges = []
        self.subdiv = None  # reference to subidiv class
        self.constrained_meshes = {} # dictionary to hold reference of points from constrained selection to mesh idx


    def check_meshes(self, input_meshes):
        list_poly_data = []
        # check if meshes are in a list
        if isinstance(input_meshes, list):
            # check if meshes are PySubdiv.Mesh or PyVistaPolydata
            for idx, mesh in enumerate(input_meshes):
                if isinstance(mesh, (PySubdiv.Mesh, pyvista.core.pointset.PolyData)):
                    # check which type, if PySubdiv.Mesh convert to PyVista.PolyData and append
                    if isinstance(mesh, PySubdiv.Mesh):
                        list_poly_data.append(mesh.model().clean())
                    # if not append directly
                    else:
                        list_poly_data.append(mesh.clean())
                else:
                    raise TypeError(f"mesh at index {idx} is type {type(mesh)}. "
                                    f"Types should be PySubdiv.Mesh or Pyvista.PolyData")
        # check correct type of passed single mesh
        else:
            if isinstance(input_meshes, (PySubdiv.Mesh, pyvista.core.pointset.PolyData)):
                if isinstance(input_meshes, (PySubdiv.Mesh, pyvista.core.pointset.PolyData)):
                    # check which type, if PySubdiv.Mesh convert to PyVista.PolyData and append
                    if isinstance(input_meshes, PySubdiv.Mesh):
                        list_poly_data.append(input_meshes.model().clean())
                    # if not append directly
                    else:
                        list_poly_data.append(input_meshes.clean())
            else:
                raise TypeError(f"mesh is type {type(input_meshes)}. "
                                f"Types should be PySubdiv.Mesh or Pyvista.PolyData")
        self.input_meshes = list_poly_data
        return self.input_meshes

    def mesh_visibility(self, idx_mesh=None):
        button = self.visibility_button_dict[idx_mesh]
        if button.isChecked():
            # check if mesh is selected; if selected set state of the button to False
            if idx_mesh in self.selected_meshes:
                print(f"mesh_{idx_mesh} is selected, please deselect first")
                button.setChecked(False)
                # self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)

            # set mesh to invisible
            else:
                button.setToolTip('make visible')
                self.invisible_meshes.append(idx_mesh)
                self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                      name=f"mesh_{idx_mesh}", opacity=0, reset_camera=False)
                self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-points')
                self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-labels')
        # make mesh visible
        else:
            button.setToolTip('make invisible')
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
            self.plotter.add_point_labels([self.input_meshes[idx_mesh].center_of_mass()], [f"mesh_{idx_mesh}"],
                                          name=f"labels_mesh_{idx_mesh}", reset_camera=False)
            self.invisible_meshes.remove(idx_mesh)

    def select_mesh(self, idx_mesh=None):
        button = self.select_button_dict[idx_mesh]
        boundary_button = self.select_constrain_button_dict[idx_mesh]
        button_min_max = self.select_constrain_min_max_dict[idx_mesh]

        if button.isChecked():
            if self._state != States.picking_selection:
                button.setChecked(False)
                print('Selection is not possible')
                return

            # check if mesh is invisible; if invisible set state of button to false
            if idx_mesh in self.invisible_meshes:
                print(f"mesh_{idx_mesh} is invisible and cannot be selected")
                button.setChecked(False)
            # select the current mesh
            else:
                button.setToolTip('deselect')
                self.selected_meshes.append(idx_mesh)
                self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                      name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
                # set button to constrain the selection only on the boundary vertices/min max points
                boundary_button.show()
                boundary_button.setEnabled(True)
                boundary_button.setToolTip('constrain to boundary')
                button_min_max.show()
                button_min_max.setEnabled(True)
                button_min_max.setToolTip('constrain to extrema')

        # deselect the mesh
        else:
            button.setToolTip('select')
            # check if boundary vertices actor is active -> remove
            if f"constrained_mesh{idx_mesh}" in self.plotter.renderer.actors:
                self.plotter.remove_actor(f"constrained_mesh{idx_mesh}")

            # deactivate the button for boundary vertices/min max when the mesh is deselected
            boundary_button.hide()
            boundary_button.setEnabled(False)
            boundary_button.setChecked(False)
            boundary_button.setToolTip('constrain to boundary')
            button_min_max.hide()
            button_min_max.setChecked(False)
            button_min_max.setEnabled(False)
            button_min_max.setToolTip('constrain to extrema')

            # make mesh unpickable and deselect
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
            # set state of the button to false
            button.setChecked(False)
            self.selected_meshes.remove(idx_mesh)

    def point_to_selection(self, mesh, point_idx):
        if self._state == States.picking_selection:
            # Get coordinates of the point
            point = mesh.points[point_idx]
            print(self.input_meshes)
            if mesh in self.input_meshes:
                mesh_idx = self.input_meshes.index(mesh)
            else:
                mesh_idx = list(self.constrained_meshes.keys())[list(self.constrained_meshes.values()).index(mesh)]

            # Append to selection

            if mesh_idx not in self.selection_idx:
                self.selection_idx[mesh_idx] = {}

            if point_idx not in self.selection_idx[mesh_idx]:
                self.selection.append(point)
                idx_in_selection = len(self.selection) - 1
                self.selection_idx[mesh_idx][point_idx] = idx_in_selection
            self.selection_mesh += pv.PolyData(point)
            self.plotter.add_points(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                    name='selection',
                                    reset_camera=False)

        elif self._state == States.deleting_selection:
            self.selection.pop(point_idx)
            # delete index pairs from the selection index dictionary -> finding the idx in mesh deleting
            # the index in self.selection array
            for index_pairs in self.selection_idx.values():
                for idx_original, idx_in_list in index_pairs.items():
                    if idx_in_list == point_idx:
                        index_pairs.pop(idx_original)
                        break
                # decrementing the indices above the deleted one
                for idx_original, idx_in_list in index_pairs.items():
                    if idx_in_list > point_idx:
                        index_pairs[idx_original] = idx_in_list - 1

            if len(self.selection) == 0:
                # clear the selection
                self.selection_idx.clear()
                self.selection_mesh = pv.PolyData()
                self.plotter.remove_actor('selection')

                # if all points deleted set state back to picking
                self.switch_picking_deleting(True)
                button_idx = self.button_widgets_utils['button_picking_deleting']
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(True)
                return

            self.selection_mesh = pv.PolyData(self.selection)
            self.plotter.add_points(self.selection_mesh, color='purple', pickable=True, point_size=10, name='selection',
                                    reset_camera=False)
        elif self._state == States.mesh_subdivision:
            self.control_points = mesh
            mesh_idx = self.input_meshes.index(mesh)
            self.selected_meshes = [mesh_idx]
            if mesh_idx not in self.selection_idx:
                self.selection_idx[mesh_idx] = {}

            if point_idx in self.selection_idx[mesh_idx]:
                self.selection.pop(point_idx)
                self.selection_idx[mesh_idx].pop(point_idx)
            else:
                self.selection.append(self.control_points.points[point_idx])
                if point_idx not in self.selection_idx[mesh_idx]:
                    idx_in_selection = len(self.selection) - 1
                    self.selection_idx[mesh_idx][point_idx] = idx_in_selection

            scalar = np.zeros(mesh.n_points)
            if len(self.selection) != 0:
                scalar[np.array(list(self.selection_idx[mesh_idx].keys()))] = 1
            self.plotter.add_mesh(mesh, scalars=scalar, cmap=['green', 'purple'], pickable=False,
                                  name=f"control_points", reset_camera=False,
                                  style='points', render_points_as_spheres=True, point_size=10, show_scalar_bar=False)
        else:
            return

    def point_to_polygon(self, mesh, point_idx):
        if point_idx in self.selection_polygon_idx:
            return
        self.selection_polygon_idx.append(point_idx)
        self.selection_polygon_mesh.faces = np.array([len(self.selection_polygon_idx), ] + self.selection_polygon_idx)
        if self.selection_polygon_mesh.faces[0] > 2:
            self.selection_polygon_mesh = self.selection_polygon_mesh.compute_normals(cell_normals=True,
                                                                                      point_normals=False)
            # normal of polygon face
            normal = self.selection_polygon_mesh["Normals"][0]
            # center point of the polygon
            center = self.selection_polygon_mesh.cell_centers().points[0]
            # dot product from normal and vector to current camera position
            dot_product = np.dot(normal, (center - self.plotter.camera_position[0]))
            if dot_product > 0:
                color = 'lightblue'
            else:
                color = 'yellow'

            self.plotter.add_mesh(self.selection_polygon_mesh, color=color, pickable=False, point_size=15,
                                  name='selection_polygon', show_edges=True, reset_camera=False)

    def triangulate_selection(self, check):
        if self._state == States.mesh_decimation:
            print('Mesh decimation activated')
            return
        if check:
            # check if vertices are selected
            if len(self.selection) < 3:
                print('Less than 3 vertices in the selection')
                return

            # user_input = input("Polygon for cutting? ") or False

            msg = 'Use polygon for cutting?'
            user_input = easygui.ynbox(msg)

            if user_input:
                if "polygon_button" in self.button_widgets_utils:
                    idx_polygon_button = self.button_widgets_utils['polygon_button']
                    self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetVisibility(True)
                    self.plotter.button_widgets[idx_polygon_button].On()
                else:
                    self.plotter.add_checkbox_button_widget(self.selection_to_polygon, position=(80, 10))
                    self.button_widgets_utils['polygon_button'] = len(self.plotter.button_widgets) - 1
                self.plotter.add_text('abort', font_size=6, position=(16, 70), name="triangulate_abort_text")

            else:
                if len(self.selection) == 3:
                    mesh = pv.PolyData(np.array(self.selection), faces=[3, 0, 1, 2])
                else:
                    mesh = pv.PolyData(np.array(self.selection)).delaunay_2d(edge_source=self.selection_polygon_mesh)
                idx_mesh = len(self.input_meshes)  # index in self.input_meshes
                # add the mesh to the plotter
                self.add_meshes_to_input_and_plotter(mesh, idx_mesh, new_mesh=True)
                button_idx = self.button_widgets_utils['triangulation_button']
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
        else:
            if self._state == States.polygon_cutting:
                self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, pickable_window=False,
                                                  show_point=False)
                self.selection_polygon_mesh = pv.PolyData()
                self.plotter.remove_actor('selection_polygon')
                self.selection_polygon_idx.clear()
                self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                      name='selection', reset_camera=False)

                idx_polygon_button = self.button_widgets_utils['polygon_button']
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetVisibility(False)
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetState(False)
                self.plotter.button_widgets[idx_polygon_button].Off()
                self.plotter.remove_actor("triangulate_abort_text")
                self.set_state(States.picking_selection)
            else:
                button_idx = self.button_widgets_utils['triangulation_button']
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)

                idx_polygon_button = self.button_widgets_utils['polygon_button']
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetVisibility(False)
                self.plotter.button_widgets[idx_polygon_button].GetRepresentation().SetState(False)
                self.plotter.button_widgets[idx_polygon_button].Off()
                self.plotter.remove_actor("triangulate_abort_text")

    def constrain_selection_to_boundary(self, idx_mesh):
        button = self.select_constrain_button_dict[idx_mesh]
        if button.isChecked():
            button.setToolTip('remove constrains')
            # uncheck button for min max vertices
            button_min_max = self.select_constrain_min_max_dict[idx_mesh]
            button_min_max.setToolTip('constrain to extrema')
            button_min_max.setChecked(False)

            # extract vertices on the boundary and make mesh unpickable only points on the boundary are pickable
            mesh = self.input_meshes[idx_mesh]
            boundary = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                  manifold_edges=False, feature_edges=False)

            # threshold for boundary vertices
            threshold = QInputDialog.getInt(self, "enter threshold angle", "angle:", 0, 0, 360, 1)[0]
            connected_vertices = {vertex_idx: [] for vertex_idx in range(boundary.n_points)}
            counter = 2
            for line_idx in range(boundary.n_lines):
                line_vertex_idx_1 = boundary.lines[counter - 1]
                line_vertex_idx_2 = boundary.lines[counter]
                connected_vertices[line_vertex_idx_1].append(line_vertex_idx_2)
                connected_vertices[line_vertex_idx_2].append(line_vertex_idx_1)
                counter += 3  # two vertices per edge

            vertices_after_threshold = []
            for idx_vertex in range(boundary.n_points):
                point_1, point_2 = boundary.points[connected_vertices[idx_vertex]][:2]
                footpoint = boundary.points[idx_vertex]
                angle = calculation.angle_vertices(point_1, point_2, footpoint)
                angle = abs((angle * 180 / np.pi) - 180)
                if angle > threshold:
                    vertices_after_threshold.append(footpoint)
            boundary_vertices = np.array(vertices_after_threshold)
            self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)

            self.plotter.add_points(boundary_vertices, pickable=True, name=f"constrained_mesh{idx_mesh}",
                                    reset_camera=False)
            self.constrained_meshes[idx_mesh] = pv.PolyData(boundary_vertices)

        else:
            button.setToolTip('constrain selection to boundary vertices')
            # remove actor for vertices on the boundary and make the mesh pickable again
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
            self.plotter.remove_actor(f"constrained_mesh{idx_mesh}")
            self.constrained_meshes.pop(idx_mesh)

    def selection_to_polygon(self, check):
        if check:
            self.deselect_all_meshes(True)
            self._state = States.polygon_cutting
            self.plotter.enable_point_picking(self.point_to_polygon, use_mesh=True, pickable_window=False)
            self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=True, point_size=10, name='selection',
                                  reset_camera=False)
            self.selection_polygon_mesh.points = np.array(self.selection)
        else:
            # delauny triangulation with the polygon as mask
            mesh = pv.PolyData(np.array(self.selection)).delaunay_2d(edge_source=self.selection_polygon_mesh)
            idx_mesh = len(self.input_meshes)  # index in self.input_meshes
            # self.new_meshes_idx[idx_mesh] = len(self.new_meshes_idx.keys())

            # add the mesh to the plotter
            self.add_meshes_to_input_and_plotter(mesh, idx_mesh, new_mesh=True)

            # remove the selection polygon from plotter + clear lists
            self.plotter.remove_actor('selection_polygon')
            self.selection_polygon_idx.clear()
            self.selection_polygon_mesh = pv.PolyData()

            # deactivate the button end enable picking on meshes
            self.plotter.button_widgets[self.button_widgets_utils["polygon_button"]].Off()
            self.plotter.button_widgets[self.button_widgets_utils["polygon_button"]].GetRepresentation().SetState(False)
            self.plotter.button_widgets[
                self.button_widgets_utils["polygon_button"]].GetRepresentation().SetVisibility(False)

            # self.button_widgets_utils.pop("polygon_button", None)
            self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, pickable_window=False)
            self._state = States.picking_selection

    def selection_undo(self, check):
        if check:
            if self._state == States.picking_selection:
                if len(self.selection) == 0:
                    return

                point_idx = len(self.selection) - 1
                self.selection.pop()
                # delete index pairs from the selection index dictionary -> finding the idx in mesh deleting
                # the index in self.selection array
                for index_pairs in self.selection_idx.values():
                    for idx_original, idx_in_list in index_pairs.items():
                        if idx_in_list == point_idx:
                            index_pairs.pop(idx_original)
                            break
                    # decrementing the indices above the deleted one
                    for idx_original, idx_in_list in index_pairs.items():
                        if idx_in_list > point_idx:
                            index_pairs[idx_original] = idx_in_list - 1

                if len(self.selection) == 0:
                    self.selection_idx.clear()
                    self.selection_mesh = pv.PolyData()
                    self.plotter.remove_actor('selection')
                    return

                self.selection_mesh = pv.PolyData(np.array(self.selection))
                self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                      name='selection', reset_camera=False)

            elif self._state == States.polygon_cutting:
                if len(self.selection_polygon_idx) == 0:
                    return

                self.selection_polygon_idx.pop()
                self.selection_polygon_mesh.faces = np.array([len(self.selection_polygon_idx), ]
                                                             + self.selection_polygon_idx)

                self.plotter.add_mesh(self.selection_polygon_mesh, color='lightblue', pickable=False, point_size=15,
                                      name='selection_polygon', show_edges=True, reset_camera=False)
            else:
                return
        else:
            self.selection_undo(True)

    def switch_picking_deleting(self, picking):
        button_idx = self.button_widgets_utils['button_picking_deleting']
        button_state = self.plotter.button_widgets[button_idx].GetRepresentation().GetState()
        if self._state != States.picking_selection and self._state != States.deleting_selection:
            if button_state:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
            else:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(True)
            return
        if picking:
            self.set_state(States.picking_selection)
            self.plotter.add_text('select', font_size=6, position=(210, 60), name="selection_text")
            if len(self.selection) != 0:
                self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=False, point_size=10,
                                      name='selection', reset_camera=False)

        else:
            if len(self.selection) == 0:
                if button_state:
                    self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                else:
                    self.plotter.button_widgets[button_idx].GetRepresentation().SetState(True)
                return

            self.set_state(States.deleting_selection)
            self.plotter.add_text('delete', font_size=6, position=(210, 60), name="selection_text")
            self.deselect_all_meshes(True)
            self.plotter.add_mesh(self.selection_mesh, color='purple', pickable=True, point_size=10, name='selection',
                                  reset_camera=False)

    def add_meshes_to_input_and_plotter(self, mesh, idx_mesh, new_mesh=False, merged_mesh=False):
        if None in self.input_meshes:
            idx_mesh = self.input_meshes.index(None)
            self.input_meshes[idx_mesh] = mesh

        else:
            self.input_meshes.append(mesh)

        if new_mesh:
            if None in self.new_meshes_idx.values():
                counter = 0
                for mesh_index, new_mesh_index in self.new_meshes_idx.items():
                    if new_mesh_index is None:
                        # search for the next free slot
                        while counter in self.new_meshes_idx.values():
                            counter += 1
                        self.new_meshes_idx[idx_mesh] = counter
                        break
                    else:
                        counter += 1
            else:
                self.new_meshes_idx[idx_mesh] = len(self.new_meshes_idx.keys())

            # find boundary_points of the new mesh
            boundary = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                  manifold_edges=False, feature_edges=False)
            boundary_points = boundary.points

            edges = mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

            boundary_edges = boundary.lines.reshape((-1, 3))[:, 1:]

            edge_dict = {}

            for edge_idx, edge in enumerate(boundary_edges):
                edge_masked = boundary_points[edge]
                if tuple(edge_masked[0].tolist()) not in edge_dict:
                    edge_dict[tuple(edge_masked[0].tolist())] = {tuple(edge_masked[1].tolist())}
                else:
                    edge_dict[tuple(edge_masked[0].tolist())].update([tuple(edge_masked[1].tolist())])
                if tuple(edge_masked[1].tolist()) not in edge_dict:
                    edge_dict[tuple(edge_masked[1].tolist())] = {tuple(edge_masked[0].tolist())}
                else:
                    edge_dict[tuple(edge_masked[1].tolist())].update([tuple(edge_masked[0].tolist())])
            boundary_edges_data = np.zeros(len(edges))
            for edge_idx, edge in enumerate(edges):
                edge_masked = mesh.points[edge]
                if tuple(edge_masked[0].tolist()) in edge_dict:
                    if tuple(edge_masked[1].tolist()) in edge_dict[tuple(edge_masked[0].tolist())]:
                        boundary_edges_data[edge_idx] = 1
            vertex_dict = {}
            boundary_vertices = np.ones(mesh.n_points, dtype=int)
            for point_idx, point in enumerate(boundary_points):
                vertex_dict[tuple(point.tolist())] = point_idx
            for vrtx_idx, vrtx in enumerate(mesh.points):
                if tuple(vrtx.tolist()) not in vertex_dict:
                    boundary_vertices[vrtx_idx] = -1

            mesh.point_data['boundary_vertices'] = boundary_vertices
            mesh.field_data['boundary_edges_data'] = boundary_edges_data

            # set dynamic_vertices for the new mesh

            msg = 'Enter mesh index for dynamic vertices or cancel'
            dynamic_vertex = QInputDialog.getInt(self, "enter idx from mesh to fit to, -1 for static", "idx:", -1)[0]
            if dynamic_vertex is not None:
                if dynamic_vertex < 0:
                    dynamic_vertex = 's'
                dynamic_vertices = np.array([dynamic_vertex for _ in range(mesh.n_points)], dtype=str)
                dynamic_faces = np.array([dynamic_vertex for _ in range(mesh.n_cells)], dtype=str)
                mesh["dynamic_vertices"] = dynamic_vertices
                mesh["dynamic_faces"] = dynamic_faces
        if merged_mesh:
            if None in self.merged_meshes_idx.values():
                print(self.merged_meshes_idx)
                counter = 0
                for mesh_index, merged_mesh_index in self.merged_meshes_idx.items():
                    if merged_mesh_index is None:
                        # search for the next free slot
                        while counter in self.merged_meshes_idx.values():
                            counter += 1
                        self.merged_meshes_idx[idx_mesh] = counter
                        break
                    else:
                        counter += 1
            else:
                self.merged_meshes_idx[idx_mesh] = len(self.merged_meshes_idx.keys())

            # find non-manifold vertices of the merged mesh
            non_manifold_edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
                                                            manifold_edges=False, feature_edges=False)

            non_manifold_points = non_manifold_edges.points

            vertex_dict = {}
            non_manifold_vertices = np.ones(mesh.n_points, dtype=int)
            for point_idx, point in enumerate(non_manifold_points):
                vertex_dict[tuple(point.tolist())] = point_idx
            for vrtx_idx, vrtx in enumerate(mesh.points):
                if tuple(vrtx.tolist()) not in vertex_dict:
                    non_manifold_vertices[vrtx_idx] = -1

            mesh.point_data['non_manifold_vertices'] = non_manifold_vertices

        # add the mesh to the plotter
        self.plotter.add_mesh(mesh, color="green", show_edges=True, name=f"mesh_{idx_mesh}", pickable=False,
                              reset_camera=False)
        self.plotter.add_point_labels([mesh.center_of_mass()], [f"mesh_{idx_mesh}"], name=f"labels_mesh_{idx_mesh}",
                                      reset_camera=False)
        x_pos = 10
        offset = 50 * idx_mesh
        if new_mesh:
            x_pos = 115
            offset = 50 * self.new_meshes_idx[idx_mesh]
        if merged_mesh:
            x_pos = 220
            offset = 50 * self.merged_meshes_idx[idx_mesh]

        label = QLabel(self.frame)
        label.setText(f"mesh_{idx_mesh}")
        label.move(x_pos + 10, 50 + offset)
        label.setStyleSheet(f"background-color: dimgrey")
        label.show()
        self.mesh_text_dic[f"mesh_name{idx_mesh}"] = label

        # add buttons for making the mesh invisible
        btn = QPushButton(f"{idx_mesh}", self.plotter)
        btn.move(x_pos, 70 + offset)
        btn.setToolTip('make invisible')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.resize(15, 15)
        btn.show()
        text = btn.text()
        btn.setStyleSheet('QPushButton {background-color: green; color: blue; font-size: 1px}')
        self.visibility_button_dict[idx_mesh] = btn
        btn.clicked.connect(lambda ch, text=text: self.mesh_visibility(int(text)))

        # add buttons for selecting the mesh

        btn = QPushButton(f"{idx_mesh}", self.plotter)
        btn.move(x_pos + 20, 70 + offset)
        btn.show()
        btn.setToolTip('select')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.resize(15, 15)
        text = btn.text()
        btn.setStyleSheet('QPushButton {background-color: red; color: blue; font-size: 1px}')
        self.select_button_dict[idx_mesh] = btn
        btn.clicked.connect(lambda ch, text=text: self.select_mesh(int(text)))

        # add buttons for constrain selection to boundary

        btn = QPushButton(f"{idx_mesh}", self.plotter)
        btn.move(x_pos + 40, 70 + offset)
        btn.show()
        btn.setToolTip('constrain selection to boundary')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.resize(15, 15)
        text = btn.text()
        btn.setStyleSheet('QPushButton {background-color: blue; color: blue; font-size: 1px}')
        self.select_constrain_button_dict[idx_mesh] = btn
        btn.clicked.connect(lambda ch, text=text: self.constrain_selection_to_boundary(int(text)))
        btn.setEnabled(False)
        btn.hide()

        # add buttons for min max

        btn = QPushButton(f"{idx_mesh}", self.plotter)
        btn.move(x_pos + 60, 70 + offset)
        btn.show()
        btn.setToolTip('constrain selection to extrema')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.resize(15, 15)
        text = btn.text()
        btn.setStyleSheet('QPushButton {background-color: blue; color: blue; font-size: 1px}')
        self.select_constrain_min_max_dict[idx_mesh] = btn
        btn.clicked.connect(lambda ch, text=text: self.constrain_selection_min_max(int(text)))
        btn.setEnabled(False)
        btn.hide()

        # add buttons for deleting

        btn = QPushButton(f"{idx_mesh}", self.plotter)
        btn.move(x_pos + 80, 70 + offset)
        btn.show()
        btn.setToolTip('delete mesh from viewer')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.resize(15, 15)
        text = btn.text()
        btn.setStyleSheet('QPushButton {background-color: darkred; color: blue; font-size: 1px}')
        self.delete_button[idx_mesh] = btn
        btn.clicked.connect(lambda ch, text=text: self.delete_mesh(int(text)))
        return idx_mesh

    def deselect_all_meshes(self, check):
        # loop over selected meshes and deselect all
        if check:
            temp_list = copy.copy(self.selected_meshes)
            for idx_mesh in temp_list:
                self.select_button_dict[idx_mesh].setChecked(False)
                self.select_mesh(idx_mesh)
        else:
            self.deselect_all_meshes(True)

    def clear_selection(self, check):
        if check:
            if self._state == States.picking_selection or self._state == States.deleting_selection:
                if len(self.selection) != 0:

                    if self.mover is not None:
                        self.mover_btn['Move selection'].setChecked(False)
                        self.mover.widget_off()

                    self.selection.clear()
                    self.selection_idx.clear()
                    button_index = self.button_widgets_utils['button_picking_deleting']
                    self.plotter.button_widgets[button_index].GetRepresentation().SetState(True)
                    self.set_state(States.picking_selection)

                    # check if selection is edges
                    if len(self.selected_edges) > 0:
                        self.selection_mesh['selected'] = np.zeros(self.selection_mesh.n_lines)
                        self.selected_edges.clear()
                        self.plotter.add_mesh(self.selection_mesh, scalars='selected', cmap=['black', 'purple'],
                                              pickable=True,
                                              show_edges=True,
                                              name=f"selection", opacity=100, reset_camera=False,
                                              render_lines_as_tubes=True, line_width=10, show_scalar_bar=False)
                    else:
                        self.plotter.remove_actor('selection')
                        self.selection_mesh = pv.PolyData()

            if self._state == States.polygon_cutting:
                self.selection_polygon_idx.clear()
                self.selection_polygon_mesh = pv.PolyData()
                self.selection_polygon_mesh.points = np.array(self.selection)
                self.plotter.remove_actor('selection_polygon')

            if self._state == States.mesh_subdivision:
                self.selection.clear()
                self.selection_idx.clear()

                # check if selection is edges
                if len(self.selected_edges) > 0:
                    self.selection_mesh['selected'] = np.zeros(self.selection_mesh.n_lines)
                    self.selected_edges.clear()
                    self.plotter.add_mesh(self.selection_mesh, scalars='selected', cmap=['black', 'purple'],
                                          pickable=True,
                                          show_edges=True,
                                          name=f"selection", opacity=100, reset_camera=False,
                                          render_lines_as_tubes=True, line_width=10, show_scalar_bar=False)
                    self.subdiv.edge_selection_changed()

                if self.control_points is None:
                    return
                self.plotter.add_mesh(self.control_points, color='green', pickable=False,
                                      name=f"control_points", reset_camera=False,
                                      style='points', render_points_as_spheres=True, point_size=10,
                                      show_scalar_bar=False)
        else:
            self.clear_selection(True)

    def merge_meshes(self, check):
        if check:
            mesh_1 = self.input_meshes[int(input("Index first mesh: "))]
            mesh_2 = self.input_meshes[int(input("Index second mesh: "))]
            print(mesh_1.array_names)
            print(mesh_2.array_names)

            # self.merged_meshes_idx[idx_mesh] = len(self.merged_meshes_idx.keys())

            msg = 'Recalculate boundary vertices? '
            user_bool = easygui.ynbox(msg)
            if user_bool:
                # recalculate boundary_points of the new mesh
                merged_mesh = mesh_1.merge(mesh_2, inplace=False)
                idx_mesh = len(self.input_meshes)  # index in self.input_meshes

                boundary = merged_mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                             manifold_edges=False, feature_edges=False)
                boundary_points = boundary.points
                boundary_points_mesh_1 = mesh_1.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                      manifold_edges=False, feature_edges=False).points

                boundary_points_mesh_2 = mesh_2.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                      manifold_edges=False, feature_edges=False).points

                edges = merged_mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

                boundary_edges = boundary.lines.reshape((-1, 3))[:, 1:]
                # points_masked_non_manifold_edges = non_manifold_points[non_manifold_edges]
                # print(points_masked_non_manifold_edges)
                # print(points_masked_edges)

                edge_dict = {}

                for edge_idx, edge in enumerate(boundary_edges):
                    edge_masked = boundary_points[edge]
                    if tuple(edge_masked[0].tolist()) not in edge_dict:
                        edge_dict[tuple(edge_masked[0].tolist())] = {tuple(edge_masked[1].tolist())}
                    else:
                        edge_dict[tuple(edge_masked[0].tolist())].update([tuple(edge_masked[1].tolist())])
                    if tuple(edge_masked[1].tolist()) not in edge_dict:
                        edge_dict[tuple(edge_masked[1].tolist())] = {tuple(edge_masked[0].tolist())}
                    else:
                        edge_dict[tuple(edge_masked[1].tolist())].update([tuple(edge_masked[0].tolist())])
                boundary_edges_data = np.zeros(len(edges))
                for edge_idx, edge in enumerate(edges):
                    edge_masked = merged_mesh.points[edge]
                    if tuple(edge_masked[0].tolist()) in edge_dict:
                        if tuple(edge_masked[1].tolist()) in edge_dict[tuple(edge_masked[0].tolist())]:
                            boundary_edges_data[edge_idx] = 1

                merged_mesh.field_data['boundary_edges_data'] = boundary_edges_data

                if 'boundary_layer_vertices' in mesh_1.array_names:
                    print('2')
                    boundary_layers_old = mesh_1.points[np.nonzero(mesh_1['boundary_layer_vertices'] == 1)]
                    boundary_points_mesh_1 = np.concatenate((boundary_points_mesh_1, np.array(boundary_layers_old)))
                    boundary_points_mesh_2 = np.concatenate((boundary_points_mesh_2, np.array(boundary_layers_old)))

                if 'boundary_layer_vertices' in mesh_2.array_names:
                    print('3')
                    boundary_layers_old = mesh_2.points[np.nonzero(mesh_2['boundary_layer_vertices'] == 1)]
                    boundary_points_mesh_2 = np.concatenate((boundary_points_mesh_2, np.array(boundary_layers_old)))
                    boundary_points_mesh_1 = np.concatenate((boundary_points_mesh_1, np.array(boundary_layers_old)))

                vertex_set_mesh_1 = set()
                vertex_set_mesh_2 = set()

                for point in boundary_points_mesh_1:
                    vertex_set_mesh_1.add(tuple(point))
                for point in boundary_points_mesh_2:
                    vertex_set_mesh_2.add(tuple(point))
                vertex_intersection = vertex_set_mesh_1.intersection(vertex_set_mesh_2)

                vertex_dict = {}
                boundary_vertices = np.ones(merged_mesh.n_points, dtype=int)
                boundary_layer_vertices = np.zeros(merged_mesh.n_points, dtype=int)

                for point_idx, point in enumerate(boundary_points):
                    vertex_dict[tuple(point.tolist())] = point_idx
                for vrtx_idx, vrtx in enumerate(merged_mesh.points):
                    if tuple(vrtx.tolist()) not in vertex_dict:
                        boundary_vertices[vrtx_idx] = -1
                        if tuple(vrtx.tolist()) in vertex_intersection:
                            boundary_layer_vertices[vrtx_idx] = 1

                merged_mesh.point_data['boundary_vertices'] = boundary_vertices
                merged_mesh.point_data['boundary_layer_vertices'] = boundary_layer_vertices
            else:

                if 'boundary_layer_vertices' in mesh_1.array_names and 'boundary_layer_vertices' in mesh_2.array_names:
                    merged_mesh = mesh_1.merge(mesh_2, inplace=False)
                elif 'boundary_layer_vertices' in mesh_1.array_names:
                    mesh_2.point_data['boundary_layer_vertices'] = np.zeros(mesh_2.n_points, dtype=int)
                    merged_mesh = mesh_1.merge(mesh_2, inplace=False)
                elif 'boundary_layer_vertices' in mesh_2.array_names:
                    mesh_1.point_data['boundary_layer_vertices'] = np.zeros(mesh_1.n_points, dtype=int)
                    merged_mesh = mesh_2.merge(mesh_1, inplace=False)
                else:
                    merged_mesh = mesh_1.merge(mesh_2, inplace=False)

                idx_mesh = len(self.input_meshes)  # index in self.input_meshes

                if 'boundary_edges_data' in mesh_1.array_names and 'boundary_edges_data' in mesh_2.array_names:
                    edges_1 = mesh_1.extract_all_edges()
                    edges_2 = mesh_2.extract_all_edges()
                    edges_mesh_1 = edges_1.lines.reshape((-1, 3))[:, 1:]
                    edges_mesh_2 = edges_2.lines.reshape((-1, 3))[:, 1:]
                    boundary_edges_mesh_1 = edges_mesh_1[np.nonzero(mesh_1['boundary_edges_data'] == 1)]
                    boundary_edges_mesh_2 = edges_mesh_2[np.nonzero(mesh_2['boundary_edges_data'] == 1)]
                    print(np.max(boundary_edges_mesh_1))
                    print(np.max(boundary_edges_mesh_2))
                    boundary_points_mesh_1 = edges_1.points.reshape((-1, 3))
                    print(len(boundary_points_mesh_1))

                    boundary_points_mesh_2 = edges_2.points.reshape((-1, 3))
                    print(len(boundary_points_mesh_2))

                else:
                    # extract boundary edges and points of mesh_1 and mesh_2
                    boundary_mesh_1 = mesh_1.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                   manifold_edges=False, feature_edges=False)

                    boundary_edges_mesh_1 = boundary_mesh_1.lines.reshape((-1, 3))[:, 1:]

                    boundary_mesh_2 = mesh_2.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                                   manifold_edges=False, feature_edges=False)

                    boundary_edges_mesh_2 = boundary_mesh_2.lines.reshape((-1, 3))[:, 1:]

                    boundary_points_mesh_1 = boundary_mesh_1.points
                    boundary_points_mesh_2 = boundary_mesh_2.points

                # extract all edges of merged mesh and concatenate boundary edges and points of mesh_1 and mesh_2
                edges = merged_mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]

                boundary_edges_conc = [[boundary_edges_mesh_1] + [boundary_edges_mesh_2]]

                boundary_points_conc = [boundary_points_mesh_1] + [boundary_points_mesh_2]

                edges_dict = [{}, {}]

                # loop over the boundary edges of mesh_1 and mesh_2, mask the edges with the vertex position
                # and create a dictionary with vertex position per edge
                for mesh_idx, boundary_edges in enumerate(boundary_edges_conc[0]):
                    print(boundary_edges)
                    for edge in boundary_edges:
                        print(mesh_idx)
                        print(edge)

                        edge_masked = boundary_points_conc[mesh_idx][edge]
                        print(edge_masked)
                        if tuple(edge_masked[0].tolist()) not in edges_dict[mesh_idx]:
                            edges_dict[mesh_idx][tuple(edge_masked[0].tolist())] = {tuple(edge_masked[1].tolist())}
                        else:
                            edges_dict[mesh_idx][tuple(edge_masked[0].tolist())].update(
                                [tuple(edge_masked[1].tolist())])
                        if tuple(edge_masked[1].tolist()) not in edges_dict[mesh_idx]:
                            edges_dict[mesh_idx][tuple(edge_masked[1].tolist())] = {tuple(edge_masked[0].tolist())}
                        else:
                            edges_dict[mesh_idx][tuple(edge_masked[1].tolist())].update(
                                [tuple(edge_masked[0].tolist())])
                print(edges_dict)
                # loop over edges of the merged mesh and find edges where both meshes share a boundary edge
                boundary_edges_data = np.zeros(len(edges))
                for edge_idx, edge in enumerate(edges):
                    edge_masked = merged_mesh.points[edge]
                    if tuple(edge_masked[0].tolist()) in edges_dict[0] and tuple(edge_masked[0].tolist()) in edges_dict[
                        1]:
                        if tuple(edge_masked[1].tolist()) in edges_dict[0][tuple(edge_masked[0].tolist())] and \
                                tuple(edge_masked[1].tolist()) in edges_dict[1][tuple(edge_masked[0].tolist())]:
                            boundary_edges_data[edge_idx] = 1

                merged_mesh.field_data['boundary_edges_data'] = boundary_edges_data

            self.add_meshes_to_input_and_plotter(merged_mesh, idx_mesh, merged_mesh=True)
        else:
            self.merge_meshes(True)

    def pick_faces(self, check):
        if check:
            self.clear_selection(True)
            self.plotter.enable_cell_picking(callback=self.faces_picker, through=False)
        else:
            self.selected_faces.clear()
            if 'face_selection' in self.plotter.renderer.actors:
                self.plotter.remove_actor('face_selection')
            self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, show_point=False)

    def pick_edges(self, text):
        edge_btn = self.edges_btn[text]

        if edge_btn.isChecked():

            if len(self.selected_meshes) == 1:
                # get index of mesh and clear rest of the selections, select the mesh again afterwards
                idx_mesh = self.selected_meshes[0]
                if self._state == States.mesh_subdivision:
                    pass
                else:
                    self.deselect_all_meshes(True)
                    button_select = self.select_button_dict[idx_mesh]
                    button_select.setChecked(True)
                    self.select_mesh(idx_mesh)

                self.clear_selection(True)
                self.selected_faces.clear()
                self.selection_mesh = self.input_meshes[idx_mesh].extract_all_edges()
                self.selection_mesh['selected'] = np.zeros(self.selection_mesh.n_lines)

                if self._state == States.mesh_subdivision:
                    opacity = self.subdiv.transparency
                else:
                    opacity = 100

                self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=False, show_edges=True,
                                      name=f"mesh_{idx_mesh}", opacity=opacity, reset_camera=False)

                self.plotter.add_mesh(self.selection_mesh, color='black', pickable=True, show_edges=True,
                                      name=f"selection", opacity=100, reset_camera=False,
                                      render_lines_as_tubes=True, line_width=10)

                vtk_custom_widget.enable_cell_picking(self.plotter, callback=self.edges_picker, through=False,
                                                      show=False)
            else:
                edge_btn.setChecked(False)
                return
        else:

            idx_mesh = self.selected_meshes[0]

            self.clear_selection(True)

            if self._state == States.mesh_subdivision:
                opacity = self.subdiv.transparency
            else:
                opacity = 100
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=opacity, reset_camera=False)

            self.selection_mesh = pv.PolyData()
            self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True)
            self.plotter.remove_actor('selection')

            if self.mover is not None:
                self.mover_btn['Move selection'].setChecked(False)
                self.mover.widget_off()

    def edges_picker(self, passed_edges):
        if passed_edges is None:
            return
        edge_id = passed_edges['original_cell_ids']
        point_ids = passed_edges['vtkOriginalPointIds']
        mesh_idx = self.selected_meshes[0]
        if mesh_idx not in self.selection_idx:
            self.selection_idx[mesh_idx] = {}
        if edge_id in self.selected_edges:
            self.selection_mesh['selected'][edge_id] = 0
            self.selected_edges.remove(edge_id)
            for point_idx in point_ids:
                self.selection_idx[mesh_idx][point_idx][point_idx] -= 1
                if self.selection_idx[mesh_idx][point_idx][point_idx] == 0:
                    to_delete = []
                    for i, point in enumerate(self.selection):
                        if {tuple(point.tolist())} == {tuple(self.selection_mesh.points[point_idx].tolist())}:
                            to_delete.append(i)
                    self.selection_idx[mesh_idx].pop(point_idx)
                    self.selection = [point for i, point in enumerate(self.selection) if i not in to_delete]

        else:
            self.selection_mesh['selected'][edge_id] = 1
            self.selected_edges.append(edge_id)
            self.selection.extend(passed_edges.points)
            for point_idx in point_ids:
                if point_idx not in self.selection_idx[mesh_idx]:
                    self.selection_idx[mesh_idx][point_idx] = {point_idx: 1}
                else:
                    self.selection_idx[mesh_idx][point_idx][point_idx] += 1

        self.plotter.add_mesh(self.selection_mesh, scalars='selected', cmap=['black', 'purple'], pickable=True,
                              show_edges=True,
                              name=f"selection", opacity=100, reset_camera=False,
                              render_lines_as_tubes=True, line_width=10, show_scalar_bar=False)

        if self._state == States.mesh_subdivision:
            self.subdiv.edge_selection_changed()

    def faces_picker(self, passed_triangle):
        if len(self.selected_meshes) > 1:
            return
        idx_mesh = self.selected_meshes[0]
        print(idx_mesh)
        # for face_idx in passed_triangle["original_cell_ids"]:
        #     print(face_idx)

        if idx_mesh in self.selected_faces:
            self.selected_faces[idx_mesh].update(tuple(passed_triangle["original_cell_ids"].tolist()))
        else:
            self.selected_faces[idx_mesh] = set(tuple(passed_triangle["original_cell_ids"].tolist()))

        face_selection = self.input_meshes[idx_mesh].extract_cells(np.array(list(self.selected_faces[idx_mesh])))
        self.plotter.add_mesh(face_selection.extract_all_edges(), color='yellow', pickable=False, show_edges=True,
                              name=f"face_selection", opacity=100, reset_camera=False)

    def delete_faces(self):
        for idx_mesh in self.selected_faces:
            print(list(self.selected_faces[idx_mesh]))
            self.input_meshes[idx_mesh].remove_cells(np.array(list(self.selected_faces[idx_mesh])), inplace=True)
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='green', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
        self.pick_faces(False)

    def mesh_decimation(self, check):
        if check:
            button_idx = self.button_widgets_utils['mesh_decimation']
            if self._state != States.picking_selection and self._state != States.deleting_selection:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                return
            if len(self.selected_meshes) < 1:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                print('Please select a mesh')
                return
            elif len(self.selected_meshes) > 1:
                self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)
                print('Please select only one mesh')
                return

            self.clear_selection(True)
            self.set_state(States.mesh_decimation)
            temp_mesh = self.input_meshes[self.selected_meshes[0]]
            self.deselect_all_meshes(True)
            pv.save_meshio("temp_mesh.obj", temp_mesh)

            # convert to meshlab
            # define a helper function for the button to call the set_percent_decimation function
            def set_reduction_helper(check_1):
                if check_1:
                    msg = 'Set reduction factor: (float)'
                    # user_input = float(input("Set reduction factor: ")) or 0.1
                    successful_input = False
                    while not successful_input:
                        user_input = easygui.enterbox(msg, default='0.1')
                        if user_input is None:
                            return
                        else:
                            try:
                                user_input = float(user_input)
                                successful_input = True
                            except ValueError:
                                print("Input is not a float, please try again or cancel")

                    idx_slider = self.button_widgets_utils['mesh_decimation_slider']
                    self.plotter.slider_widgets[idx_slider].GetRepresentation().SetValue(user_input)
                    self.set_percent_decimation(user_input)
                else:
                    set_reduction_helper(True)

            if 'mesh_decimation_slider' not in self.button_widgets_utils:
                # if called first time initialize the slider and buttons
                self.plotter.add_slider_widget(self.set_percent_decimation, rng=(0.0, 1.0))
                self.button_widgets_utils['mesh_decimation_slider'] = len(self.plotter.slider_widgets) - 1

                self.plotter.add_checkbox_button_widget(set_reduction_helper, position=(800, 10), color_on='orange',
                                                        color_off='grey')
                self.plotter.add_text('set reduction ', font_size=6, position=(780, 60), name='set_reduction_button')
                self.button_widgets_utils['mesh_decimation_reduction'] = len(self.plotter.button_widgets) - 1

                self.plotter.add_checkbox_button_widget(self.abort_decimation, position=(900, 10), color_on='orange',
                                                        color_off='orange')
                self.plotter.add_text('abort decimation', font_size=6, position=(880, 60),
                                      name='abort_decimation_button')
                self.button_widgets_utils['mesh_abort_decimation'] = len(self.plotter.button_widgets) - 1

            else:

                # activate slider and button

                slider_idx = self.button_widgets_utils['mesh_decimation_slider']
                self.plotter.slider_widgets[slider_idx].GetRepresentation().SetVisibility(True)
                self.plotter.slider_widgets[slider_idx].GetRepresentation().SetValue(0.5)
                self.plotter.slider_widgets[slider_idx].On()

                self.plotter.add_text('set reduction ', font_size=6, position=(780, 60), name='set_reduction_button')
                button_idx = self.button_widgets_utils['mesh_decimation_reduction']

                self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOn()
                self.plotter.button_widgets[button_idx].On()

                self.plotter.add_text('abort decimation', font_size=6, position=(880, 60),
                                      name='abort_decimation_button')
                button_idx = self.button_widgets_utils['mesh_abort_decimation']
                self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOn()
                self.plotter.button_widgets[button_idx].On()
                self.set_percent_decimation(0.5)

            # Button for setting reduction

        else:
            try:
                # testing if it is possible to subdivide the mesh -> decimation may produce topology which causes
                # problems during subdivision or if the mesh is to large memory allocation error can occur
                mesh_for_subdivision = files.read("temp_mesh_2.obj")
                mesh_for_subdivision.subdivide()
            except MemoryError as m:
                raise MemoryError(m, 'Subdivision might fail on the mesh, try to decrease the number of vertices')
            except IndexError as i:
                raise IndexError(i, 'Decimation might have created bad topology, please try to increase the reduction'
                                    'factor when possible.')
            else:
                # add the mesh to the input meshes and change the state
                # if None in self.input_meshes:
                #     idx_mesh = self.input_meshes.index(None)
                # else:

                idx_mesh = len(self.input_meshes)

                # self.new_meshes_idx[idx_mesh] = len(self.new_meshes_idx.keys())
                self.add_meshes_to_input_and_plotter(self.decimated_mesh, idx_mesh, new_mesh=True)

                # deactivate slider, buttons and delete temp. meshes
                self.abort_decimation(True)

    def set_percent_decimation(self, reduction_percent=0.5):
        # initilize pymeshlab set for performing decimation
        ms = pymeshlab.MeshSet()
        # load the tempory file
        ms.load_new_mesh("temp_mesh.obj")
        # decimation with the passed reduction
        ms.simplification_quadric_edge_collapse_decimation(targetperc=reduction_percent)
        # save the decimated mesh temporary
        ms.save_current_mesh("temp_mesh_2.obj")
        # read again with pyvista
        self.decimated_mesh = pv.read("temp_mesh_2.obj")
        # plot the mesh
        self.plotter.add_mesh(self.decimated_mesh, color='orange', name='decimated_mesh', show_edges=True,
                              reset_camera=False)

    def abort_decimation(self, check):
        if check:
            # set state back to picking
            self.set_state(States.picking_selection)
            # deactivate the slider and button
            slider_idx = self.button_widgets_utils['mesh_decimation_slider']
            self.plotter.slider_widgets[slider_idx].GetRepresentation().SetVisibility(False)
            self.plotter.slider_widgets[slider_idx].Off()

            self.plotter.remove_actor('set_reduction_button')
            button_idx = self.button_widgets_utils['mesh_decimation_reduction']
            self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOff()
            self.plotter.button_widgets[button_idx].Off()

            self.plotter.remove_actor('abort_decimation_button')
            button_idx = self.button_widgets_utils['mesh_abort_decimation']
            self.plotter.button_widgets[button_idx].GetRepresentation().VisibilityOff()
            self.plotter.button_widgets[button_idx].Off()

            button_idx = self.button_widgets_utils['mesh_decimation']
            self.plotter.button_widgets[button_idx].GetRepresentation().SetState(False)

            # remove additional mesh and delete temporary files
            self.decimated_mesh = pv.PolyData()
            self.plotter.remove_actor('decimated_mesh')
            os.remove("temp_mesh.obj")
            os.remove("temp_mesh_2.obj")
        else:
            self.abort_decimation(True)

    def constrain_selection_min_max(self, idx_mesh):
        button = self.select_constrain_min_max_dict[idx_mesh]
        if button.isChecked():
            button.setToolTip('remove constrain')
            mesh = self.input_meshes[idx_mesh]

            button_boundary_constrain = self.select_constrain_button_dict[idx_mesh]
            button_boundary_constrain.setToolTip('constrain to boundary')
            button_boundary_constrain.setChecked(False)

            # button_boundary_idx = self.button_widgets_boundary[f"button_boundary_{idx_mesh}"]
            # self.plotter.button_widgets[button_boundary_idx].GetRepresentation().SetState(False)

            # deactivate mesh for picking
            self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)

            if 'extrema' in mesh.array_names:
                extrema_points_idx = np.nonzero(mesh["extrema"] == 1)
                if len(extrema_points_idx) == 0:
                    # no min max found; make whole mesh pickable again
                    self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                          name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
                else:
                    extrema_points = mesh.points[extrema_points_idx]
                    self.plotter.add_points(extrema_points, pickable=True, name=f"constrained_mesh{idx_mesh}",
                                            reset_camera=False)
            else:
                extrema = np.zeros(mesh.n_points)
                points = mesh.points
                edges = mesh.extract_all_edges().lines.reshape((-1, 3))[:, 1:]
                edges_extracted = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                             feature_edges=False,
                                                             manifold_edges=False)
                boundary_vertices_set = set()
                boundary_vertices = edges_extracted.points.tolist()
                for vertex in boundary_vertices:
                    boundary_vertices_set.add(tuple(vertex))
                vertex_vertex_dictionary = data_structure.vertex_vertex_dictionary(points, edges)
                local_minima = []
                local_maxima = []

                for i, vertex in enumerate(points):
                    if not tuple(vertex) in boundary_vertices_set:
                        is_extrema = []
                        vertex_z = vertex[2]
                        for vertex_idx in vertex_vertex_dictionary[i]:
                            neighbor_z = points[vertex_idx][2]
                            if neighbor_z > vertex_z:
                                is_extrema.append('Min')
                            elif neighbor_z < vertex_z:
                                is_extrema.append('Max')
                        is_extrema = np.array(is_extrema)
                        if all(is_extrema == 'Min'):
                            local_minima.append(i)
                        elif all(is_extrema == 'Max'):
                            local_maxima.append(i)

                    else:
                        is_extrema = []
                        vertex_z = vertex[2]
                        for vertex_idx in vertex_vertex_dictionary[i]:
                            if tuple(points[vertex_idx]) in boundary_vertices_set:
                                neighbor_z = points[vertex_idx][2]
                                if neighbor_z > vertex_z:
                                    is_extrema.append('Min')
                                elif neighbor_z < vertex_z:
                                    is_extrema.append('Max')
                        is_extrema = np.array(is_extrema)
                        if len(is_extrema) > 0:
                            if all(is_extrema == 'Min'):
                                local_minima.append(i)
                            elif all(is_extrema == 'Max'):
                                local_maxima.append(i)

                local_extrema = local_minima + local_maxima
                extrema[local_extrema] = 1
                mesh.point_data['extrema'] = extrema
                if len(local_extrema) == 0:
                    # no min max found; make whole mesh pickable again
                    self.plotter.add_mesh(mesh, color='red', pickable=False, show_edges=True,
                                          name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
                else:
                    extrema_points = points[local_extrema]
                    self.plotter.add_points(extrema_points, pickable=True, name=f"constrained_mesh{idx_mesh}",
                                            reset_camera=False)
                    self.constrained_meshes[idx_mesh] = pv.PolyData(extrema_points)
        else:
            button.setToolTip('constrain to extrema')
            # remove actor for vertices on min max and make the mesh pickable again
            self.plotter.add_mesh(self.input_meshes[idx_mesh], color='red', pickable=True, show_edges=True,
                                  name=f"mesh_{idx_mesh}", opacity=100, reset_camera=False)
            self.plotter.remove_actor(f"constrained_mesh{idx_mesh}")
            self.constrained_meshes.pop(idx_mesh)

    def delete_mesh(self, idx_mesh):
        delete_button = self.delete_button[idx_mesh]
        if idx_mesh in self.invisible_meshes:
            print(f"mesh_{idx_mesh} is invisible, please make visible again to delete")
            delete_button.setChecked(False)
            return
        if delete_button.isChecked():

            # message box for confimation with save, cancel, and ok
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Delete mesh from viewer")
            msg.setText("Do you want to delete the mesh?")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Save | QMessageBox.Cancel)
            return_value = msg.exec()
            if return_value == 4194304:
                delete_button.setChecked(False)
                return
            elif return_value == 2048:
                self.save_mesh(mesh_idx=idx_mesh)

            if idx_mesh in self.selected_meshes:
                self.select_button_dict[idx_mesh].setChecked(False)
                self.select_mesh(idx_mesh)
            self.input_meshes[idx_mesh] = None
            if idx_mesh in self.new_meshes_idx:
                self.new_meshes_idx[idx_mesh] = None
            if idx_mesh in self.merged_meshes_idx:
                self.merged_meshes_idx[idx_mesh] = None
            if idx_mesh in self.input_meshes_data:
                self.input_meshes_data.pop(idx_mesh)

            # delete button and remove the mesh
            self.mesh_text_dic[f"mesh_name{idx_mesh}"].deleteLater()
            self.mesh_text_dic[f"mesh_name{idx_mesh}"] = None
            self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-points')
            self.plotter.remove_actor(f'labels_mesh_{idx_mesh}-labels')
            self.delete_button[idx_mesh] = None
            delete_button.deleteLater()
            self.visibility_button_dict[idx_mesh].deleteLater()
            self.visibility_button_dict[idx_mesh] = None
            self.select_button_dict[idx_mesh].deleteLater()
            self.select_button_dict[idx_mesh] = None
            self.select_constrain_min_max_dict[idx_mesh].deleteLater()
            self.select_constrain_min_max_dict[idx_mesh] = None
            self.select_constrain_button_dict[idx_mesh].deleteLater()
            self.select_constrain_button_dict[idx_mesh] = None
            self.plotter.remove_actor(f"mesh_{idx_mesh}")
        else:
            self.delete_mesh(True)

    def save_mesh(self, mesh_idx=None):
        if mesh_idx is None or mesh_idx is False:
            title = 'Save a mesh'
            msg = 'select a mesh for saving'
            choices = [f"mesh_{i}" for i, mesh in enumerate(self.input_meshes) if mesh is not None]
            if len(choices) == 0:
                return
            elif len(choices) == 1:
                choice = choices[0]
            else:
                choice = easygui.choicebox(msg, title, choices)
            if choice is None:
                return
            mesh_idx = int(choice.split('_')[1])
        else:
            choice = f"mesh_{mesh_idx}"

        msg = 'save mesh as .obj file'
        filename = easygui.filesavebox(msg, default=f"{choice}.obj", filetypes="\\*.obj")
        if filename is None:
            return
        if not filename.endswith('.obj'):
            filename += ".obj"

        pv.save_meshio(filename, self.input_meshes[mesh_idx])
        temp_pysubdiv = files.read(filename)
        try:
            temp_pysubdiv.data['dynamic_vertices'] = self.input_meshes[mesh_idx]['dynamic_vertices']
            temp_pysubdiv.data['fitting_method'] = 'dynamic_faces'
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['dynamic_faces'] = self.input_meshes[mesh_idx]['dynamic_faces']
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['boundary_vertices'] = self.input_meshes[mesh_idx]['boundary_vertices']
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['non_manifold_vertices'] = self.input_meshes[mesh_idx]['non_manifold_vertices']
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['extrema'] = self.input_meshes[mesh_idx]['extrema']
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['boundary_layer_vertices'] = self.input_meshes[mesh_idx]['boundary_layer_vertices']
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['boundary_edges_data'] = self.input_meshes[mesh_idx]['boundary_edges_data']
        except KeyError as k:
            print(k)
        try:
            temp_pysubdiv.data['deviation'] = self.input_meshes[mesh_idx]['deviation']
        except KeyError as k:
            print(k)

        temp_pysubdiv.save_data(filename[:-4] + "Data")

    def load_mesh(self):
        msg = 'Select a .obj File for loading'
        file = easygui.fileopenbox(msg, filetypes="*.obj")
        if file is None:
            return
        elif not file.endswith('.obj'):
            print('The file is not in .obj format, please try again')
        else:
            mesh = pv.read(file)

            file_path = file.rsplit("/", 1)
            path_txt = [file for file in sorted(os.listdir(file_path[0])) if file.endswith(".pkl")]

            idx_mesh = len(self.input_meshes)
            idx_mesh = self.add_meshes_to_input_and_plotter(mesh, idx_mesh)

            msg = 'Load additional data for the file?'
            user_bool = easygui.ynbox(msg)

            if user_bool:

                if file_path[1][:-4] + "Data.pkl" in path_txt:
                    file_data = file_path[0] + "/" + file_path[1][:-4] + "Data.pkl"
                else:
                    msg = 'Select data file (.pkl) for the mesh'
                    file_data = easygui.fileopenbox(msg, filetypes="*.pkl")
                if file_data is None:
                    pass
                else:
                    while not file_data.endswith('pkl'):
                        msg = 'The file is not in .pkl format, do you want to try again?'
                        user_bool = easygui.ynbox(msg)
                        if not user_bool:
                            break
                        else:
                            msg = 'Select data file (.pkl) for the mesh'
                            file_data = easygui.fileopenbox(msg, filetypes="*.pkl")
                    temp_mesh = files.read(file)
                    temp_mesh.load_data(file_data)
                    if 'dynamic_vertices' in temp_mesh.data:
                        self.input_meshes[idx_mesh].point_data['dynamic_vertices'] = temp_mesh.data[
                            'dynamic_vertices']
                    if 'dynamic_faces' in temp_mesh.data:
                        self.input_meshes[idx_mesh].cell_data['dynamic_faces'] = temp_mesh.data['dynamic_faces']
                    if 'boundary_vertices' in temp_mesh.data:
                        self.input_meshes[idx_mesh].point_data['boundary_vertices'] = temp_mesh.data[
                            'boundary_vertices']
                    if 'boundary_layer_vertices' in temp_mesh.data:
                        self.input_meshes[idx_mesh].point_data['boundary_layer_vertices'] = temp_mesh.data[
                            'boundary_layer_vertices']

                    if 'boundary_edges_data' in temp_mesh.data:
                        self.input_meshes[idx_mesh].field_data['boundary_edges_data'] = temp_mesh.data[
                            'boundary_edges_data']

                    if 'deviation' in temp_mesh.data:
                        self.input_meshes[idx_mesh].point_data['deviation'] = temp_mesh.data[
                            'deviation']

                    # if 'non_manifold_vertices' in temp_mesh.data:
                    #     self.input_meshes[idx_mesh].cell_data['non_manifold_vertices'] = temp_mesh.data[
                    #         'non_manifold_vertices']
                    if 'extrema' in temp_mesh.data:
                        self.input_meshes[idx_mesh].point_data['extrema'] = temp_mesh.data['extrema']
                    self.input_meshes_data[idx_mesh] = file_data

    def visualize_data_array(self):
        self.deselect_all_meshes(True)
        window = Second(self.input_meshes, self.plotter, self)

    def render_selected_mesh_as_spheres(self):
        if len(self.selected_meshes) == 1:
            if 'dynamic_vertices' in self.input_meshes[self.selected_meshes[0]].array_names:
                scalar = self.input_meshes[self.selected_meshes[0]]['dynamic_vertices']
            else:
                scalar = None
            self.plotter.add_mesh(self.input_meshes[self.selected_meshes[0]], name=f"mesh_{self.selected_meshes[0]}",
                                  reset_camera=False, pickable=False, style='points', scalars=scalar,
                                  render_points_as_spheres=True, show_scalar_bar=False)

    def define_volumes(self):
        volumes_module = DefineVolumes(self.selected_faces, self.input_meshes, self)

    def main(self):
        offset = 0

        for idx, mesh in enumerate(self.input_meshes):
            self.plotter.add_mesh(mesh, color='green', pickable=False, show_edges=True, name=f"mesh_{idx}",
                                  reset_camera=False)
            self.plotter.add_point_labels([mesh.center_of_mass()], [f"mesh_{idx}"], name=f"labels_mesh_{idx}",
                                          reset_camera=False)

            label = QLabel(self.plotter)
            label.setText(f"mesh_{idx}")
            label.move(10, 50 + offset)
            label.setStyleSheet(f"background-color: dimgrey")
            self.mesh_text_dic[f"mesh_name{idx}"] = label

            # add buttons for making the mesh invisible
            btn = QPushButton(f"{idx}", self.plotter)
            btn.move(10, 70 + offset)
            btn.setToolTip('make invisible')
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.resize(15, 15)
            text = btn.text()
            btn.setStyleSheet('QPushButton {background-color: green; color: blue; font-size: 1px}')
            self.visibility_button_dict[idx] = btn
            btn.clicked.connect(lambda ch, text=text: self.mesh_visibility(int(text)))

            # add buttons for selecting the mesh

            btn = QPushButton(f"{idx}", self.plotter)
            btn.move(30, 70 + offset)
            btn.setToolTip('select')
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.resize(15, 15)
            text = btn.text()
            btn.setStyleSheet('QPushButton {background-color: red; color: blue; font-size: 1px}')
            self.select_button_dict[idx] = btn
            btn.clicked.connect(lambda ch, text=text: self.select_mesh(int(text)))

            # add buttons for constrain selection to boundary

            btn = QPushButton(f"{idx}", self.plotter)
            btn.move(50, 70 + offset)
            btn.setToolTip('constrain selection to boundary')
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.resize(15, 15)
            text = btn.text()
            btn.setStyleSheet('QPushButton {background-color: blue; color: blue; font-size: 1px}')
            self.select_constrain_button_dict[idx] = btn
            btn.clicked.connect(lambda ch, text=text: self.constrain_selection_to_boundary(int(text)))
            btn.setEnabled(False)
            btn.hide()

            # add buttons for min max

            btn = QPushButton(f"{idx}", self.plotter)
            btn.move(70, 70 + offset)
            btn.setToolTip('constrain selection to extrema')
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.resize(15, 15)
            text = btn.text()
            btn.setStyleSheet('QPushButton {background-color: blue; color: blue; font-size: 1px}')
            self.select_constrain_min_max_dict[idx] = btn
            btn.clicked.connect(lambda ch, text=text: self.constrain_selection_min_max(int(text)))
            btn.setEnabled(False)
            btn.hide()

            # add buttons for deleting

            btn = QPushButton(f"{idx}", self.plotter)
            btn.move(90, 70 + offset)
            btn.setToolTip('delete mesh from viewer')
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.resize(15, 15)
            text = btn.text()
            btn.setStyleSheet('QPushButton {background-color: darkred; color: blue; font-size: 1px}')
            self.delete_button[idx] = btn
            btn.clicked.connect(lambda ch, text=text: self.delete_mesh(int(text)))

            # vtk_custom_widget.checkbox_button_widget(self.plotter, callback=self.delete_mesh, idx_mesh=idx,
            #                                          position=(90, 680 - offset), size=15, border_size=2,
            #                                          color_on='red')
            #
            # self.button_widgets[f"button_deletion{idx}"] = len(self.plotter.button_widgets) - 1

            offset += 50

        self.plotter.add_checkbox_button_widget(self.triangulate_selection, color_on='purple', color_off='purple')
        self.plotter.add_text('triangulation', font_size=6, position=(6, 60), name="triangulate_text")
        self.button_widgets_utils['triangulation_button'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.selection_undo, position=(150, 10))
        self.button_widgets_utils['polygon_selection_undo'] = len(self.plotter.button_widgets) - 1

        # button switch between selection and deletion
        self.plotter.add_checkbox_button_widget(self.switch_picking_deleting, position=(200, 10), value=True)
        self.plotter.add_text('select', font_size=6, position=(210, 60), name="selection_text")
        self.button_widgets_utils['button_picking_deleting'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.deselect_all_meshes, position=(300, 10))
        self.button_widgets_utils['deselect_all_meshes'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.clear_selection, position=(400, 10))
        self.button_widgets_utils['clear_selection'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.merge_meshes, position=(500, 10), color_on='yellow',
                                                color_off='yellow')
        self.button_widgets_utils['merge_meshes'] = len(self.plotter.button_widgets) - 1

        self.plotter.add_checkbox_button_widget(self.pick_faces, position=(600, 10), color_on='purple',
                                                color_off='grey')
        self.plotter.add_text('select faces', font_size=6, position=(600, 60), name="delete_faces_text")
        self.button_widgets_utils['delete_faces'] = len(self.plotter.button_widgets) - 1

        # Buttons + slider for mesh decimation
        self.plotter.add_checkbox_button_widget(self.mesh_decimation, position=(700, 10), color_on='orange',
                                                color_off='grey')
        self.plotter.add_text('mesh decimation', font_size=6, position=(680, 60))
        self.button_widgets_utils['mesh_decimation'] = len(self.plotter.button_widgets) - 1

        # add button for moving selection
        btn_move = QPushButton('Move selection', self.plotter)
        btn_move.move(1000, 10)
        btn_move.setToolTip('move selection')
        btn_move.resize(120, 40)
        btn_move.setCheckable(True)
        btn_move.setChecked(False)
        text = btn_move.text()
        self.mover_btn[text] = btn_move
        btn_move.clicked.connect(lambda ch, text=text: self.move_selection(text))

        # add button for selecting edges
        btn_edges = QPushButton('Select Edges', self.plotter)
        btn_edges.move(1150, 10)
        btn_edges.setToolTip('Select Edges')
        btn_edges.resize(120, 40)
        btn_edges.setCheckable(True)
        btn_edges.setChecked(False)
        text = btn_edges.text()
        self.edges_btn[text] = btn_edges
        btn_edges.clicked.connect(lambda ch, text=text: self.pick_edges(text))

        # # Button for saving a cotrol_mesh
        # self.plotter.add_checkbox_button_widget(self.save_mesh, position=(1000, 10), color_on='lightblue',
        #                                         color_off='lightblue')
        # self.plotter.add_text('save mesh', font_size=6, position=(990, 60))
        # self.button_widgets_utils['save_mesh'] = len(self.plotter.button_widgets) - 1
        # # Button for loading a mesh
        # self.plotter.add_checkbox_button_widget(self.load_mesh, position=(1100, 10), color_on='green',
        #                                         color_off='green')
        # self.plotter.add_text('load mesh', font_size=6, position=(1090, 60))
        # self.button_widgets_utils['load_mesh'] = len(self.plotter.button_widgets) - 1

        # # Button for showing curvature
        # self.plotter.add_checkbox_button_widget(self.show_curvature, position=(1200, 10), color_on='green',
        #                                         color_off='green')
        # self.plotter.add_text('show curvature', font_size=6, position=(1190, 60))
        # self.button_widgets_utils['show curvature'] = len(self.plotter.button_widgets) - 1

        self.plotter.enable_point_picking(self.point_to_selection, use_mesh=True, pickable_window=False,
                                          show_point=False)
        self.plotter.add_camera_orientation_widget()
        self.set_state(States.picking_selection)
        self.plotter.isometric_view_interactive()
        self.plotter.show()

    def move_selection(self, text):
        btn = self.mover_btn[text]
        if btn.isChecked():
            if len(self.selection) == 0:
                btn.setChecked(False)
                return

            if self._state == States.mesh_subdivision:
                # print(np.array(self.selection))
                # points = self.input_meshes[self.selected_meshes[0]].points[np.array(self.selection)]

                # print(points)
                # print(self.input_meshes, self.selection_mesh, self.subdiv)
                self.mover = vtk_custom_widget.add_position_widget(self.plotter, self.selection, self.selection_idx,
                                                                   self.input_meshes, self.selection_mesh, self.subdiv)
            else:
                self.mover = vtk_custom_widget.add_position_widget(self.plotter, self.selection, self.selection_idx,
                                                                   self.input_meshes, self.selection_mesh)
        else:
            self.mover.widget_off()

    def subdivide(self):
        self.deselect_all_meshes(True)
        self.clear_selection(True)
        self.set_state(States.mesh_subdivision)
        self.subdiv = Subdivide(self.input_meshes, self.input_meshes_data, self.plotter, self.control_points,
                                self.selected_meshes, self.selected_edges, self)

    def set_state(self, state):
        self._state = state

    @staticmethod
    def find_face_index(picked_face_passed, vertex_dict, faces_dict):
        picked_face_vertices = picked_face_passed.points
        vertex_indices_from_mesh = []
        for vert in picked_face_vertices:
            if tuple(vert.tolist()) not in vertex_dict:
                return
            vertex_indices_from_mesh.append(vertex_dict[tuple(vert.tolist())])
        face_idx = faces_dict[tuple(sorted(vertex_indices_from_mesh))]
        return face_idx


class Second(MainWindow):
    def __init__(self, input_meshes, plotter, parent=None):
        super(Second, self).__init__(parent)
        self.plotter = plotter
        self.input_meshes = input_meshes
        self.scalar = None
        self.cb = QComboBox(self)
        self.cb.setGeometry(120, 14, 300, 25)
        self.cb.currentIndexChanged.connect(self.selection_change)

        self.cb_data_array = QComboBox(self)
        self.cb_data_array.setGeometry(120, 64, 300, 25)

        self.cb_data_faces = QComboBox(self)
        self.cb_data_faces.setGeometry(120, 114, 300, 25)

        self.label = QLabel(self)
        self.label.setText("select a mesh: ")
        self.label.move(10, 10)

        self.label_data = QLabel(self)
        self.label_data.setText("face data: ")
        self.label_data.move(10, 110)

        self.label_point_data = QLabel(self)
        self.label_point_data.setText("point data: ")
        self.label_point_data.move(10, 64)

        for mesh_idx, mesh in enumerate(self.input_meshes):
            if mesh is not None:
                self.cb.addItem(f"mesh_{mesh_idx}")

        self.setFixedSize(500, 400)
        self.setWindowTitle("Select mesh and property")

        self.cb_data_faces.currentIndexChanged.connect(self.set_scalars_faces)
        self.cb_data_array.currentIndexChanged.connect(self.set_scalars_points)

        self.selected_mesh = None

        self.show()
        self.cb.show()
        self.cb_data_array.show()
        self.cb_data_faces.show()

        if self.cb.count() == 1:
            self.selection_change(0)

    def selection_change(self, i):
        self.cb_data_array.clear()
        self.cb_data_faces.clear()
        if len(self.plotter.scalar_bars.keys()) > 0:
            self.plotter.remove_scalar_bar()
        self.selected_mesh = int(self.cb.itemText(i)[-1])
        for data_name in self.input_meshes[int(self.cb.itemText(i)[-1])].point_data.keys():
            self.cb_data_array.addItem(data_name)
        for data_faces in self.input_meshes[int(self.cb.itemText(i)[-1])].cell_data.keys():
            self.cb_data_faces.addItem(data_faces)

    def set_scalars_points(self, i):
        if self.selected_mesh is None:
            return
        if i == -1:
            return

        mesh = self.input_meshes[self.selected_mesh]
        self.scalar = self.cb_data_array.itemText(i)
        self.plotter.add_mesh(mesh, scalars=self.scalar, show_edges=True, name=f"mesh_{self.selected_mesh}",
                              reset_camera=False, pickable=False)

    def set_scalars_faces(self, i):
        if self.selected_mesh is None:
            return
        if i == -1:
            return
        mesh = self.input_meshes[self.selected_mesh]
        self.scalar = self.cb_data_faces.itemText(i)

        self.plotter.add_mesh(mesh, scalars=self.scalar, show_edges=True, name=f"mesh_{self.selected_mesh}",
                              reset_camera=False, pickable=False)


class Subdivide(MainWindow):
    def __init__(self, input_meshes, input_meshes_data, plotter, control_points, selected_meshes, selected_edges,
                 parent=None):
        super(Subdivide, self).__init__(parent)
        self.plotter = plotter
        self.input_meshes = input_meshes
        self.pysub_mesh = None
        self.input_meshes_data = input_meshes_data
        self.control_points = control_points
        self.selected_meshes = selected_meshes
        self.selected_edges = selected_edges
        self.iteration = 1
        self.transparency = 0.2

        self.cb = QComboBox(self)
        self.label = QLabel(self)
        self.label.setText("select a mesh: ")
        self.label.move(10, 10)

        self.label_slider = QLabel(self)
        self.label_slider.setText("transparency \n of mesh: ")
        self.label_slider.move(10, 60)

        self.trans_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.trans_slider.setGeometry(120, 64, 300, 25)
        self.trans_slider.setMinimum(0)
        self.trans_slider.setMaximum(10)
        self.trans_slider.setValue(2)
        self.trans_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.trans_slider.setTickInterval(5)
        self.trans_slider.valueChanged.connect(self.value_change)

        self.label_iteration = QLabel(self)
        self.label_iteration.setText("subdivision \n iteration: ")
        self.label_iteration.move(10, 110)

        self.iteration_input_field = QtWidgets.QLineEdit(self)
        self.iteration_input_field.setValidator(QIntValidator())
        self.iteration_input_field.setGeometry(120, 114, 50, 25)
        self.iteration_input_field.editingFinished.connect(self.change_subdiv_iterations)

        self.csv_input = QtWidgets.QDoubleSpinBox(self)

        # self.csv_input.setValidator(QDoubleValidator(0.0, 10, 2, notation=QDoubleValidator.StandardNotation))
        self.csv_input.setGeometry(120, 164, 50, 25)
        self.csv_input.editingFinished.connect(self.set_crease_values)
        self.csv_input.setMinimum(0.0)
        self.csv_input.setMaximum(1.0)
        self.csv_input.setSingleStep(0.1)
        self.csv_label = QLabel(self)
        self.csv_label.setText("crease value: ")
        self.csv_label.move(10, 160)

        self.edge_selection_changed()

        for mesh_idx, mesh in enumerate(self.input_meshes):
            if mesh is not None:
                self.cb.addItem(f"mesh_{mesh_idx}")

        self.cb.setGeometry(120, 14, 300, 25)
        self.setFixedSize(500, 400)
        self.setWindowTitle("Select mesh to subdivide")
        self.cb.currentIndexChanged.connect(self.selection_change)
        self.show()
        self.cb.show()

        if self.cb.count() == 1:
            self.selection_change(0)

    def selection_change(self, i):
        if len(self.selected_meshes) > 0:
            self.selected_meshes.clear()
        self.selected_mesh = int(self.cb.itemText(i)[-1])
        self.selected_meshes.append(self.selected_mesh)
        mesh = self.input_meshes[self.selected_mesh]
        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:]
        self.pysub_mesh = PySubdiv.Mesh(vertices, faces)
        if self.selected_mesh in self.input_meshes_data:
            self.pysub_mesh.load_data(self.input_meshes_data[self.selected_mesh])

        self.control_points = mesh
        self.plotter.add_mesh(mesh, color="green", name=f"control_points",
                              reset_camera=False, pickable=False, style='points', render_points_as_spheres=True,
                              point_size=10)
        self.plotter.add_mesh(mesh, color="red", show_edges=True, name=f"mesh_{self.selected_mesh}",
                              reset_camera=False, pickable=True, opacity=0.20)

        self.plotter.add_mesh(self.pysub_mesh.subdivide(1).model(), color="green", show_edges=True,
                              name=f"mesh_subdiv{self.selected_mesh}", reset_camera=False, pickable=False)

    def value_change(self):
        self.transparency = self.trans_slider.value() / 10
        mesh = self.input_meshes[self.selected_mesh]
        self.plotter.add_mesh(mesh, color="red", show_edges=True, name=f"mesh_{self.selected_mesh}",
                              reset_camera=False, pickable=True, opacity=self.transparency)

    def edge_selection_changed(self):
        if len(self.selected_edges) == 0:
            self.csv_input.setValue(0.0)
        else:
            mean_csv = np.mean(self.pysub_mesh.creases[np.array(self.selected_edges).flatten()])
            self.csv_input.setValue(mean_csv)

    def change_subdiv_iterations(self):
        self.iteration = int(self.iteration_input_field.text())
        if self.iteration >= 4:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText('High number of iterations may result in long compute times or memory allocation error')
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            retval = msg.exec()
            if retval == 4194304:
                return

        self.plotter.add_mesh(self.pysub_mesh.subdivide(self.iteration).model(), color="green", show_edges=True,
                              name=f"mesh_subdiv{self.selected_mesh}", reset_camera=False, pickable=False)

    def set_crease_values(self):

        if len(self.selected_edges) == 0:
            self.csv_input.setValue(0.00)
            return
        csv = self.csv_input.value()
        self.pysub_mesh.set_crease(csv, np.array(self.selected_edges).flatten())
        self.update_subdiv()

    def update_subdiv(self, selection_idx=None):
        if selection_idx is not None:
            mesh = self.input_meshes[self.selected_mesh]
            vertices = mesh.points
            # faces = mesh.faces.reshape((-1, 4))[:, 1:]
            for vert_idx in selection_idx.keys():
                self.pysub_mesh.vertices[vert_idx] = vertices[vert_idx]
        self.plotter.add_mesh(self.pysub_mesh.subdivide(self.iteration).model(), color="green", show_edges=True,
                              name=f"mesh_subdiv{self.selected_mesh}", reset_camera=False, pickable=False)


class DefineVolumes(MainWindow):
    def __init__(self, selected_faces, input_meshes, parent=None, ):
        super(DefineVolumes, self).__init__(parent)
        self.setWindowTitle("Define Volumes")
        self.cb = QComboBox(self)
        self.cb.setGeometry(120, 14, 300, 25)
        self.cb.currentIndexChanged.connect(self.selection_change)
        self.selected_mesh = None
        self.selected_faces = selected_faces
        self.input_meshes = input_meshes
        self.volumes = []

        self.label = QLabel(self)
        self.label.setText("select a mesh: ")
        self.label.move(10, 10)

        self.list_label = QLabel(self)
        self.list_label.setText("select a\nvolume: ")
        self.list_label.move(10, 60)
        self.listWidget = QListWidget(self)
        self.listWidget.setGeometry(120, 64, 250, 200)

        self.btn_add = QPushButton("+", self)
        self.btn_add.setGeometry(380, 64, 25, 25)
        self.btn_add.clicked.connect(lambda: self.add_volume(self.btn_add.text()))

        self.btn_subst = QPushButton("-", self)
        self.btn_subst.setGeometry(410, 64, 25, 25)
        self.btn_subst.clicked.connect(lambda: self.add_volume(self.btn_subst.text()))

        for mesh_idx, mesh in enumerate(self.input_meshes):
            if mesh is not None:
                self.cb.addItem(f"mesh_{mesh_idx}")

        if self.cb.count() == 1:
            self.selection_change(0)

        self.show()
        self.cb.show()
        self.btn_add.show()

        self.setFixedSize(500, 400)

    def selection_change(self, i):
        # if len(self.plotter.scalar_bars.keys()) > 0:
        #    self.plotter.remove_scalar_bar()
        self.selected_mesh = int(self.cb.itemText(i)[-1])

    def add_volume(self, text):
        print(text)
        volumes_in_list = [int(self.listWidget.item(i).text()) for i in range(self.listWidget.count())]
        self.volumes = sorted(self.volumes)
        if text == "+":
            self.listWidget.addItem(f"{len(self.volumes)}")
            self.volumes.append(len(self.volumes))
        else:
            if len(self.volumes) == 0:
                return
            idx = self.volumes.pop()
            self.listWidget.takeItem(idx)
            # if len(self.volumes) == 0:
            #     self.listWidget.takeItem(0)

        # print(len(self.volumes))
        # for idx_list, idx_volume in enumerate(self.volumes):
        #     print(idx_list)
        #     if idx_list not in volumes_in_list:
        #         self.listWidget.addItem(f"{idx_list}")
        #
        #         #
        #         # if idx_list != idx_volume:
        #         #     self.listWidget.addItem(f"{idx_list}")
        #         #     self.volumes.append(idx_list)
        #         #     break
        #         # else:
        #         #     self.listWidget.addItem(f"{idx_volume+1}")
        #         #     self.volumes.append(idx_volume+1)
        #         #     break
        print(self.volumes)
