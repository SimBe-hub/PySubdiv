from pysubdiv.main.data import files

control_cage = files.read("../controlCage/FaultDomainControlCageOptimized.obj")
control_cage.load_data("../controlCage/FaultDomainControlCageOptimizedData")
original_meshes = [files.read('../meshes/gemp_mesh_0_scaled.obj'),
                   files.read('../meshes/gemp_mesh_1_cut.obj'), files.read('../meshes/gemp_mesh_1_cut_2.obj'),
                   files.read('../meshes/gemp_mesh_2_cut_1.obj'), files.read('../meshes/gemp_mesh_2_cut_2.obj'),
                   files.read('../meshes/gemp_mesh_3_cut_1.obj'), files.read('../meshes/gemp_mesh_3_cut_2.obj'),
                   files.read('../meshes/gemp_mesh_4_cut_1.obj'), files.read('../meshes/gemp_mesh_4_cut_2.obj')]
appr_mesh = control_cage.visualize_mesh_interactive(2, original_meshes)
appr_mesh.save_mesh("FaultDomainApproximated.obj")

