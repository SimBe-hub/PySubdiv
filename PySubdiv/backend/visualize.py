import copy

import pyvista as pv
import numpy as np
import pyvista.core.pointset
from PySubdiv.backend import optimization


def create_model(faces, vertices):
    """
    Creates pyvista.PolyData object, which can be used for plotting.

    Parameters
    ----------
    faces: (n,3) or (n, 4) int
          Indexes of vertices making up the faces
    vertices: (n, 3) float
          Points in space

    Returns
    ---------
    pyvista.PolyData
        pyvista object
    """

    number_points_faces = np.ones(faces.shape[0], dtype=int) * faces.shape[1]
    faces_points = np.hstack(np.column_stack((number_points_faces, faces)))
    return pv.PolyData(vertices, faces_points)


def print_model(pysubdiv_mesh, return_plotter=False, colors=None):
    """
    Converts pysubdiv mesh to  pyvista.PolyData object and plots the object

    Parameters
    ----------
    pysubdiv_mesh: PySubdiv mesh obejct

    Returns
    ---------
    pyvista.PolyData
        pyvista object
    """
    print(colors)
    #pysubdiv_mesh.recalculate_normals()
    #model = create_model(pysubdiv_mesh.faces, pysubdiv_mesh.vertices)
    model = pysubdiv_mesh.model()
    model['label'] = np.array([i for i in range(model.n_points)])
    p = pv.Plotter()
    p.enable_depth_peeling()
    if "dynamic_faces" in pysubdiv_mesh.data:
        model.cell_arrays["opacity"] = np.where(pysubdiv_mesh.data["dynamic_faces"] == "s", 0, 1)
        if colors is not None:
            p.add_mesh(model, scalars=pysubdiv_mesh.data["dynamic_faces"], cmap=colors, show_edges=True,
                       scalar_bar_args={"color" : "black"})
        else:
            p.add_mesh(model, scalars=pysubdiv_mesh.data["dynamic_faces"], use_transparency=False, show_edges=True,
                       scalar_bar_args={"color": "black"})

    else:
        p.add_mesh(model, color='green', use_transparency=False, show_edges=True)
    #p.set_background("royalblue", top="aliceblue")
    p.set_background("white")
    p.isometric_view_interactive()
    p.add_camera_orientation_widget()
    p.show_grid(color="black")
    #p.show_bounds(all_edges=True, color="black")
    factor = p.length * 0.05

    def show_normals(check):
        if check:
            arrows = pv.Arrow()
            face_centroids = pysubdiv_mesh.face_centroids
            face_centroids_model = pv.PolyData(face_centroids)
            face_centroids_model['face_normals'] = pysubdiv_mesh.face_normals

            face_normals = face_centroids_model.glyph(orient="face_normals", geom=arrows, factor=factor)
            p.add_mesh(face_normals, show_scalar_bar=False, color='black', name='face_normals')

        else:
            p.remove_actor('face_normals')

    def show_vertex_normals(check):
        if check:
            arrows = pv.Arrow()
            mesh_vertices = pysubdiv_mesh.vertices
            mesh_vertices_poly_data = pv.PolyData(mesh_vertices)
            mesh_vertices_poly_data['vertex_normals'] = pysubdiv_mesh.vertex_normals
            vertex_normals = mesh_vertices_poly_data.glyph(orient="vertex_normals", geom=arrows, factor=factor)
            p.add_mesh(vertex_normals, show_scalar_bar=False, color='black', name='vertex_normals')
        else:
            p.remove_actor('vertex_normals')

    def show_vertex_index(check):
        if check:
            p.add_point_labels(model, "label", name="vertex_index")
        else:
            p.remove_actor("vertex_index-points")
            p.remove_actor("vertex_index-labels")

    def show_edge_index(check):
        if check:
            # find mid points of edges to plot the label at
            edges_mid_point = []
            for edge in pysubdiv_mesh.edges:
                edges_mid_point.append((pysubdiv_mesh.vertices[edge[0]] + pysubdiv_mesh.vertices[edge[1]]) / 2)
            edges_mid_point = np.array(edges_mid_point)
            # prepare label array with edge index and crease sharpness value
            edge_labels = []
            for i in range(len(pysubdiv_mesh.edges)):
                edge_labels.append(f"{i}: {pysubdiv_mesh.creases[i][0]}")
            p.add_point_labels(edges_mid_point, edge_labels, name='edge_index')
        else:
            p.remove_actor("edge_index-points")
            p.remove_actor("edge_index-labels")

    def show_static_faces(check):
        if check:
            if colors is None:
                p.add_mesh(model, scalars=pysubdiv_mesh.data["dynamic_faces"], opacity="opacity", show_edges=True,
                           scalar_bar_args={"color" : "black"})
            else:
                scalars = np.where(pysubdiv_mesh.data["dynamic_faces"] == "s", 4, pysubdiv_mesh.data["dynamic_faces"])
                scalars = scalars.astype(np.int)
                print(scalars)
                p.add_mesh(model, scalars=scalars, cmap=colors, opacity="opacity", show_edges=True,
                           scalar_bar_args={"color": "black"})

        else:
            if colors is None:
                p.add_mesh(model, scalars=pysubdiv_mesh.data["dynamic_faces"], use_transparency=False, show_edges=True,
                           scalar_bar_args={"color" : "black"})
            else:
                p.add_mesh(model, scalars=pysubdiv_mesh.data["dynamic_faces"], use_transparency=False, show_edges=True,
                           scalar_bar_args={"color": "black"}, cmap=colors)


    if "dynamic_faces" in pysubdiv_mesh.data:
        p.add_checkbox_button_widget(show_static_faces, position=(10, 300), color_on='orange', size=35)
        p.add_text('hide static faces', position=(60, 303), font_size=14, color='black')

    p.add_checkbox_button_widget(show_vertex_normals, position=(10, 260), color_on='lightblue', size=35)
    p.add_text('show vertex normals', position=(60, 263), font_size=14, color='black')
    p.add_checkbox_button_widget(show_normals, position=(10, 220), color_on='lightblue', size=35)
    p.add_text('show face normals', position=(60, 223), font_size=14, color='black')
    p.add_checkbox_button_widget(show_vertex_index, position=(10, 180), color_on='green', size=35)
    p.add_text('show vertex index', position=(60, 183), font_size=14, color='black',
               name='vertex_index')
    p.add_checkbox_button_widget(show_edge_index, position=(10, 140), color_on='green', size=35)
    p.add_text('show edge index', position=(60, 143), font_size=14, color='black',
               name='edge_index')

    if return_plotter:
        return p
    else:

        p.show()
        return model


def visualize_subdivision(mesh, iteration, additional_meshes=None):
    p = pv.Plotter()
    model_mesh = mesh.model()
    model_subdivided_mesh = mesh.subdivide(iteration).model()
    point_ids = []  # list to hold point indices
    selection = []  # list to hold vertices forming edge
    edges_with_crease = {}  # dict to hold vertices forming edge where a crease is set
    selection_edge_idx = []  # list to hold edge idx
    point_for_adding_vertex = []  # list to store point for adding

    p.add_mesh(model_mesh, color='green', opacity=0.5, use_transparency=True, show_edges=True, name='control_mesh')
    p.add_mesh(model_subdivided_mesh, color='green', use_transparency=False, show_edges=False, pickable=False,
               name='subdivided')

    radius_start = [p.length * 0.025]

    def enable_sphere_widget(check):
        def change_sphere_radius(value):
            p.clear_sphere_widgets()
            p.add_sphere_widget(update_control_cage, center=mesh.ctr_points, radius=value, color='red',
                                test_callback=False)
            radius_start[0] = value

        if check:
            for point in point_ids:
                p.remove_actor(f"point_{point}")
            for edge_idx in selection_edge_idx:
                p.remove_actor(f"selection{edge_idx}")

            selection.clear()
            point_ids.clear()
            selection_edge_idx.clear()
            p.add_slider_widget(change_sphere_radius, [0, radius_start[0]], title='change radius of spheres')
        else:
            p.clear_slider_widgets()
            p.clear_sphere_widgets()

    def update_control_cage(point, idx):
        mesh.vertices[idx] = point
        if idx in edges_with_crease:
            edge_idx = edges_with_crease[idx]
            for idx in edge_idx:
                csv = mesh.creases[idx]
                p.add_lines(model_mesh.points[mesh.edges[idx]], color='yellow',
                            width=(5 * csv) + 0.5, name=f"line{idx}")
        p.add_mesh(mesh.model(), color='green', opacity=0.5, use_transparency=True, show_edges=True,
                   name='control_mesh')

        p.add_mesh(mesh.subdivide(iteration).model(), color='green', use_transparency=False, show_edges=False,
                   pickable=False,
                   name='subdivided')

    def pick_two_points(pv_model, point_id):
        if isinstance(pv_model, pyvista.core.pointset.PolyData):
            point_ids.append(point_id)
            p.add_mesh(pv.PolyData(pv_model.points[point_id]), render_points_as_spheres=True, pickable=False,
                       point_size=radius_start[0], name=f"point_{point_id}")

        if len(point_ids) == 2:
            point_ids_sorted = np.sort(point_ids)
            idx_edge = np.nonzero((mesh.edges[:, 0] == point_ids_sorted[0]) &
                                  (mesh.edges[:, 1] == point_ids_sorted[1]))[0]

            if len(idx_edge) == 0:
                print('Selected points do not form an edge, please chose second point to be a connected vertex')
                point_ids.pop()
                p.remove_actor(f"point_{point_id}")
            else:
                selection.append(point_ids_sorted)
                selection_edge_idx.append(idx_edge)
                p.add_lines(pv_model.points[point_ids], color='white', width=7,
                            name=f"selection{idx_edge}")
                p.remove_actor([f"point_{point_ids[0]}", f"point_{point_ids[1]}"])
                point_ids.clear()

    def set_crease_values():
        if len(selection_edge_idx) == 0:
            print('No edges selected')
        else:
            print('Set crease sharpness value via console or leave empty to leave current values for the selection')
            # here is the prompt
            csv = float(input('Please enter crease sharpness value: ') or mesh.creases[selection_edge_idx])
            if csv > 1:
                csv = 1.0
            elif csv < 0:
                csv = 0
            mesh.set_crease(np.ones(len(selection_edge_idx)) * csv, np.array(selection_edge_idx).flatten())
            print(f"Crease sharpness value(s) set to: {csv}")
            p.add_mesh(mesh.subdivide(iteration).model(), color='green', use_transparency=False, show_edges=False,
                       pickable=False, name='subdivided')
            for edge_idx in selection_edge_idx:
                p.remove_actor(f"selection{edge_idx}")
                p.add_lines(model_mesh.points[mesh.edges[edge_idx][0]], color='yellow',
                            width=(5 * csv) + 0.5, name=f"line{edge_idx[0]}")

            for i in range(len(selection)):
                for vert in selection[i]:
                    edges_with_crease.setdefault(vert, []).append(selection_edge_idx[i][0])
            selection.clear()
            selection_edge_idx.clear()

    def position_for_new_vertex(click_position):
        click_position = np.array(click_position)  # convert to np.array
        direction_from_camera = click_position - p.camera_position[0]  # get direction from the camera to point
        direction_from_camera /= np.linalg.norm(direction_from_camera)  # normalize the vector
        start = click_position - 1000 * direction_from_camera  # start end for raycast
        end = click_position + 1000 * direction_from_camera
        point, ix = mesh.model().ray_trace(start, end, first_point=True)
        if len(point) > 0:
            point_for_adding_vertex.append(point)
            print(ix)
            # print(point)

            print(mesh.faces)
            mesh.add_vertex(ix[0], point)
            print(mesh.faces)
            mesh_model = mesh.model()
            p.add_mesh(mesh_model, color='green', opacity=0.5, use_transparency=True, show_edges=True,
                       name='control_mesh')

            w = p.add_mesh(pv.Sphere(radius_start[0], center=point), color='red')

        print(click_position)

    if additional_meshes is None:
        offset_for_button = 0.2
        p.add_text(
            "Press P key to pick two vertices forming an edge under the mouse to add to selection\n"
            'Press C key to set crease values for the selection of edges \n'
            'Press button to enable/disable position control of the vertices',
            color='black', font_size=14)
    else:
        if isinstance(additional_meshes, (list, tuple)):
            additional_meshes_polydata = []
            for ind_mesh in additional_meshes:
                additional_meshes_polydata.append(ind_mesh.model())
            additional_mesh = additional_meshes_polydata[0].merge(additional_meshes_polydata[1:])
        else:
            additional_mesh = additional_meshes.model()
        offset_for_button = 0
        p.add_text(
            "Press P key to pick two vertices forming an edge under the mouse to add to selection\n"
            'Press C key to set crease values for the selection of edges \n'
            'Press left button to enable/disable position control of the vertices\n'
            "Press right button to enable/disable additional meshes", color='black', font_size=14)

        def enable_disable_additional_meshes(status):
            if status:
                p.add_mesh(additional_mesh, color='blue', opacity=0.5, use_transparency=True,
                           show_edges=False, pickable=False, name='additional_mesh')
            else:
                p.remove_actor('additional_mesh')

        p.add_checkbox_button_widget(enable_disable_additional_meshes, value=False,
                                     position=(0.7 * p.window_size[0], 10))

    p.enable_point_picking(callback=pick_two_points,
                           show_message=False,
                           color='pink', point_size=20, font_size=14, use_mesh=True,
                           show_point=False, tolerance=0.1)
    p.add_checkbox_button_widget(enable_sphere_widget, color_on='red',
                                 position=((0.3 + offset_for_button) * p.window_size[0], 10))

    p.add_key_event('c', set_crease_values)
    p.set_background("royalblue", top="aliceblue")
    p.isometric_view_interactive()
    p.add_camera_orientation_widget()

    #p.track_click_position(position_for_new_vertex, side='right')

    p.show_bounds(mesh=model_mesh, grid='front', location='outer', all_edges=True)
    p.show()
    subdivided_mesh = mesh.subdivide(iteration)
    return subdivided_mesh


def visualize_distance(mesh, mesh_to_compare):
    mesh_model = mesh.model()
    distances = optimization.sdf(mesh_to_compare, mesh, return_distance=True)[1]
    mesh_model['distance'] = distances
    additional_mesh = mesh_to_compare.model()
    p = pv.Plotter()
    p.add_mesh(mesh_model, scalars='distance', use_transparency=False, show_edges=False, pickable=False,
               name='subdivided')
    p.add_mesh(additional_mesh, color='blue', opacity=0.6, use_transparency=True,
               show_edges=False, pickable=False, name='additional_mesh')
    p.set_background("royalblue", top="aliceblue")
    p.isometric_view_interactive()
    p.show_axes()
    p.show_grid()
    p.show_bounds(all_edges=True)
    p.show()


def visualize_error(mesh, meshes_to_compare, control_mesh=None, return_plotter=False, point_size=1):
    mesh_model = mesh.model()


    dist = np.array(optimization.sdf_with_meshes(meshes_to_compare, mesh)[2])

    #v = optimization.sdf_with_meshes(meshes_to_compare, mesh, return_vertices=True)

    vertices = mesh.vertices
    mesh_model['deviation'] = dist

    mesh_deleted = copy.copy(mesh_model)
    mesh_deleted.remove_cells(np.nonzero(mesh.data["dynamic_faces"] == "s"), inplace=True)

    p = pv.Plotter()
    if control_mesh is not None:
        p.add_points(control_mesh.model(), render_points_as_spheres=True, color="green", point_size=point_size)

    p.add_mesh(mesh_model, scalars="deviation", show_edges=True, cmap="seismic", name="mesh", scalar_bar_args={"color" : "black"})
    p.set_background("white")
    p.isometric_view_interactive()
    p.add_camera_orientation_widget()

    dynamic_vertices_idx = np.nonzero(mesh.data["dynamic_vertices"] != "s")

    #model.cell_arrays["opacity"] = np.where(pysubdiv_mesh.data["dynamic_faces"] == "s", 0, 1)

    def show_static_faces(check):
        if check:
            p.add_mesh(mesh_model, scalars="deviation", show_edges=True, cmap="seismic", name="mesh",
                       scalar_bar_args={"color" : "black"})
        else:
            p.add_mesh(mesh_deleted, scalars="deviation", show_edges=True, cmap="seismic", name="mesh",
                       scalar_bar_args={"color" : "black"})

    p.add_checkbox_button_widget(show_static_faces, position=(10, 300), color_on='orange', size=35)
    #p.add_text('hide static faces', position=(60, 303), font_size=14, color='black')
    #p.show_grid(color="black", font_size=25)
    p.show_bounds(color="black", font_size=25, location="outer")


    # print(f"total deviation of the meshes: "
    #       f"{np.linalg.norm((v - vertices), ord=2)}")
    # print(f"mean deviation of the meshes: {np.mean(v - vertices)}")
    # #print(f"standard deviation: {np.}")
    # print(f"maximal deviation of the meshes: {np.max((v - vertices))}")
    # print(f"Number of points: {len(mesh.vertices)}")
    # print("----------------------------------------------------------")
    # print("Error only for dynamic vertices")
    # print(f"total deviation of the meshes: "
    #       f"{np.linalg.norm((v[dynamic_vertices_idx] - vertices[dynamic_vertices_idx]), ord=2)}")
    # print(f"mean deviation of the meshes: {np.mean(v[dynamic_vertices_idx] - vertices[dynamic_vertices_idx])}")
    # print(f"maximal deviation of the meshes: {np.max((v[dynamic_vertices_idx] - vertices[dynamic_vertices_idx]))}")
    # print(f"Number of points: {len(mesh.vertices[dynamic_vertices_idx])}")

    print(f"total deviation of the meshes: {np.sum(dist)}")
    print(f"mean deviation of the meshes: {np.mean(dist)}")
    print(f"standard deviation: {np.std(dist)}")
    print(f"maximal deviation of the meshes: {np.max(dist)}")
    print(f" 25% percentile: {np.quantile(dist, 0.25)}")
    print(f" 75% percentile: {np.quantile(dist, 0.75)}")
    print(f" median: {np.median(dist)}")

    print(f"Number of points: {len(mesh.vertices)}")
    print("----------------------------------------------------------")
    print("Error only for dynamic vertices")
    print(f"total deviation of the meshes: {np.sum(dist[dynamic_vertices_idx])}")
    print(f"mean deviation of the meshes: {np.mean(dist[dynamic_vertices_idx])}")
    print(f"standard deviation: {np.std(dist[dynamic_vertices_idx])}")
    print(f"maximal deviation of the meshes: {np.max((dist[dynamic_vertices_idx]))}")
    print(f"25% percentile: {np.quantile(dist[dynamic_vertices_idx], 0.25)}")
    print(f"75% percentile: {np.quantile(dist[dynamic_vertices_idx], 0.75)}")
    print(f"median: {np.median(dist[dynamic_vertices_idx])}")
    print(f"Number of points: {len(mesh.vertices[dynamic_vertices_idx])}")

    if return_plotter:
        return p, mesh_model, mesh_deleted
    else:
        p.show()
