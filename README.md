
# PySubdiv 

#open-source python package for non-manifold subdivison surfaces algorithm

The PySubdiv library is a python-based open-source package that can apply non-manifold subdivision surfaces algorithm (Loop) on triangluar meshes with considering unique semi-sharp creases value for each edge. The implementation of semi-sharp creases in the algorithm allows to model of different varieties of structures with sharp edges.


# Quick Start

```python
import main
import files

# PySubdiv mesh objects can be created from existing faces and vertex data:

mesh = main.Mesh(
    vertices=[[-10, +30, -10], [-10, +30, 10], [20, +30, 10], [20, +30, -10], [-10, +50, -10], [-10, +50, 10],
              [20, +50, 10], [20, +50, -10], [-30, 80, -10], [-30, 80, 10], [-30, 60, 10], [-30, 60, -10],
              [-50, 50, -10], [-50, 50, 10], [-50, 30, 10], [-50, 30, -10]],

    faces=[[0, 4, 7, 3], [1, 2, 6, 5], [3, 7, 6, 2], [0, 3, 2, 1], [4, 5, 6, 7], [11, 0, 1, 10], [1, 5, 9, 10],
           [8, 9, 5, 4], [11, 8, 4, 0], [15, 11, 10, 14], [10, 9, 13, 14], [12, 13, 9, 8], [15, 12, 8, 11],
           [14, 15, 12, 13]])

# the mesh objects can be easily visualized:

mesh.visualize_mesh()

print(mesh.faces.shape)
print(mesh.faces)

# we can use the subdivision algorithms after Catmull-Clark to subdivide our mesh. A new PySubdiv mesh object will be
# returned, so better save it.

subdivided_mesh = mesh.subdivide()

# and visualize it:

subdivided_mesh.visualize_mesh()

# the number of iterations can simply be passed to the previous method:

subdivided_mesh = mesh.subdivide(iteration=2)

subdivided_mesh.visualize_mesh()

# we can create (geological-) structure to our mesh by setting individual crease value to the edges of our model
# first we can put the edges out:

print(mesh.edges_unique())

# if we already know the crease values of our edges we can pass these values to our mesh

mesh.set_crease([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 4, 5, 6, 8, 9, 13, 15, 21, 22, 23, 24, 25, 27])

# or we can simply pass a list of crease values. The crease array is than "filled" from top to bottom.

creases = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]

mesh.set_crease(creases)

subdivided_mesh = mesh.subdivide(iteration=2)
subdivided_mesh.visualize_mesh()

# By changing the crease values of particular edges we can create geological structures such as channels.
# There is also a third way of setting the creases interactively with the help of a PyVista widget:

mesh.set_crease_interactive()

subdivided_mesh = mesh.subdivide(iteration=2)
subdivided_mesh.visualize_mesh()

# We can also subdivide our mesh in an interactive PyVista widget. There we can change the control cage of our mesh and
# refine our mesh by changing the crease values. Here is also a new PySubdiv mesh object returned!

subdivided_mesh = mesh.visualize_mesh_interactive(iteration=2)

# We can also load existing meshes and load them as PySubdiv mesh objects.
# Different file format should work, at the moment .obj files are testes.
# Loaded meshes should have quadrilateral faces only.

mesh_2 = files.read('Fold.obj')

# Let's have a look and print the unique edges of the mesh:

print(mesh_2.edges_unique())
mesh_2.visualize_mesh()

# set the crease of the edges

mesh_2.set_crease([1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])

# and do the subdivision

mesh_2_subdivided = mesh_2.subdivide(iteration=3)
mesh_2_subdivided.visualize_mesh()

# Setting the crease values of the edges, we created a folded structure with a syncline and anticline.
# When looking at the crease array, we can see that most of edges' crease is set to one and only a small fraction where
# we want a smooth topology, the crease should be lowered or set to zero.
# The created meshes can be saved to vtk files, for example as .obj

subdivided_mesh.save_mesh('channel.obj')
mesh_2_subdivided.save_mesh('fold_subdivided.obj')
```


# Developers

Full stack developer : Simon Bernard (SimBe-hub)

Project manager and backend developer : s.Mohammad Moulaeifard (MohammadCGRE)

Project supervisor from RWTH Aachen university: Prof. Florian Wellmann
