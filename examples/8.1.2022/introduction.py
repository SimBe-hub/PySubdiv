from pysubdiv.main import main
from pysubdiv.main.data import files
import pyvista as pv

new_mesh_2 = files.read('../meshes/save_me.obj').simple_subdivision(1)  # loading a mesh
new_mesh_2.calculate_edge_length()
new_mesh_2.visualize_mesh()
sdsd

new_mesh_2.define_face_normals()
new_mesh_2.define_face_centroids()
new_mesh_2.define_vertex_normals()

pv_model = new_mesh_2.model()

footpoints = new_mesh_2.face_centroids
footpoints_model = pv.PolyData(footpoints)
#footpoints_model['normals'] = new_mesh_2.face_normals
new_mesh_2.recalculate_face_normals()

footpoints_model['normals_2'] = pv_model.cell_normals

footpoints_model['normals_3'] = new_mesh_2.face_normals
for i in range(len(new_mesh_2.faces)):
    print(pv_model.cell_normals[i],footpoints_model['normals_3'][i])
pv_model['normals'] = new_mesh_2.vertex_normals
p = pv.Plotter()
p.add_mesh(pv_model)
geom = pv.Arrow()
glyph = pv_model.glyph(orient="normals", geom=geom)
#glyph_2 = footpoints_model.glyph(orient='normals', scale='normals', geom=geom, factor=0.2)
glyph_3 = footpoints_model.glyph(orient='normals_2', scale='normals_2', geom=geom, factor=0.2)
glyph_4 = footpoints_model.glyph(orient='normals_3', scale='normals_3', geom=geom, factor=0.2)
#p.add_mesh(glyph, show_scalar_bar=False, lighting=False, cmap='coolwarm')
#p.add_mesh(glyph_2, show_scalar_bar=False, lighting=False, cmap='coolwarm')
p.add_mesh(glyph_3, show_scalar_bar=False, lighting=False, cmap='Dark2')
p.add_mesh(glyph_4, show_scalar_bar=False, lighting=False, cmap='coolwarm')

p.show()



sd

new_mesh = main.Mesh()  # create empty mesh object or pass vertices and faces

vertices = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]  # n,3 list n is the number of vertices
faces = [[0, 1, 2], [1, 2, 3]]   # n,3 list for triangular meshes with vertex indices  n = number of vertices


new_mesh.vertices = vertices # set vertices to the mesh
new_mesh.faces = faces # set faces to the mesh
new_mesh.edges_unique()  # create edges array for the mesh
print(new_mesh.edges)  # print edges of the mesh
print(new_mesh.vertices_connected())  # create and print matrix of connected vertices, vertices adjacency matrix or
# sometimes vertex incidence matrix
print(new_mesh.vertices_edges_incidence(to_matrix=True)) # same is possible for edges
new_mesh.faces_edges_incidence()  # and also for faces
print(new_mesh.data['faces_edges_matrix']) # all data is stored in a dictionary which we can access by giving the correct
# dictionary key here 'faces_edges_matrix'

new_mesh.print_data()  # print keys from the data array to the console

new_mesh.visualize_mesh()  # visualize the mesh

new_mesh_2 = files.read('../meshes/simple_cube.obj')  # loading a mesh
new_mesh_2.save_mesh('meshes/simple_cube.obj') # saving a mesh
new_mesh_2.save_data('meshes/simple_cube_data') # save data
new_mesh_2.load_data('meshes/simple_cube_data') # load data
new_mesh_2.set_crease([1, 0.5], [2, 1]) # first list crease values second list indices of edges
new_mesh_2.set_crease([1, 0.5, 0.3]) # set crease sharpness values from "top to bottom"


subdivided_mesh = new_mesh_2.subdivide(2)  # two times subdivision

subdivided_mesh_2 = new_mesh_2.visualize_mesh_interactive(1)  # subdivide the mesh with pyvita viewer to change csv and position of control points
new_mesh_2.visualize_mesh()
subdivided_mesh.visualize_mesh()
