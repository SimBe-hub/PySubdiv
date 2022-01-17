from pysubdiv.main.data import files
from pysubdiv.optimization import variational_minimazation

faultDomainControlCage = files.read('controlCage/FaultDomainControlCage.obj')
faultDomainControlCage.simple_subdivision().visualize_mesh()
faultDomainControlCage.load_data('controlCage/FaultDomainControlCageData.pkl')


original_meshes = [files.read('meshes/gemp_mesh_0_scaled.obj'),
                   files.read('meshes/gemp_mesh_1_cut.obj'), files.read('meshes/gemp_mesh_1_cut_2.obj'),
                   files.read('meshes/gemp_mesh_2_cut_1.obj'), files.read('meshes/gemp_mesh_2_cut_2.obj'),
                   files.read('meshes/gemp_mesh_3_cut_1.obj'), files.read('meshes/gemp_mesh_3_cut_2.obj'),
                   files.read('meshes/gemp_mesh_4_cut_1.obj'), files.read('meshes/gemp_mesh_4_cut_2.obj')]

optimizer = variational_minimazation.mesh_optimizer(faultDomainControlCage.simple_subdivision(),
                                                    original_meshes,
                                                    use_dynamic_faces=True, iterations_subdivision=1,
                                                    a_z=25)
optimizer.optimize(5, iterations_swarm=500, nr_particles=15)
cc = optimizer.control_cage
cc.save_mesh('controlCage/FaultDomainControlCageOptimized.obj')
cc.save_data('controlCage/FaultDomainControlCageOptimizedData')
cc.visualize_mesh()
cc.subdivide(2).visualize_mesh()

