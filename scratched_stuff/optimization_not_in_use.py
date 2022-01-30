def Mesh_Decimation(mesh, Reduction_factor):
    ###################################################
    # Generate PolyData mesh (triagle mesh)
    surf = mesh.model()
    print('decimated from:', len(mesh.vertices), 'vertices and', len(mesh.faces), 'faces')
    ######################################################
    # Apply Pro_Decimation Algorithm based on Pyvista
    pro_decimated = surf.decimate_pro(Reduction_factor, preserve_topology=True)
    # print(pro_decimated.n_faces)
    ######################################################
    # Convert the Result mesh to Normal Mesh for using in Py_Subdiv
    face3 = []
    for i in range(0, pro_decimated.n_faces * 4, 4):
        ee = np.array([pro_decimated.faces[i + 1], pro_decimated.faces[i + 2], pro_decimated.faces[i + 3]])
        face3.append(ee)
    # print(np.array(face3).shape)
    Decimated_mesh = main.Mesh(pro_decimated.points, face3)
    print('to:', len(Decimated_mesh.vertices), 'vertices and', len(Decimated_mesh.faces), 'faces')
    print('Reductionfactor: ', 1 - round(len(Decimated_mesh.vertices) / len(mesh.vertices), 2))

    ###############################################################
    return Decimated_mesh


##################################################################################
def signed_distance(Original_mesh, Approximated_mesh):
    Poly_Original1 = Generate_PolyDataMesh(Original_mesh)
    Poly_Approximated1 = Generate_PolyDataMesh(Approximated_mesh)

    Poly_Approximated1_Normals = Poly_Approximated1.compute_normals(point_normals=True, cell_normals=False,
                                                                    flip_normals=True,
                                                                    auto_orient_normals=True)
    #  print(Poly_Approximated1_Normals.points)
    Poly_Approximated1_Normals["distances"] = np.empty(Poly_Approximated1.n_points)
    Distance = np.empty(Poly_Approximated1.n_points)
    Contact_position_store = []

    for i in range(Poly_Approximated1_Normals.n_points):
        p = Poly_Approximated1_Normals.points[i]  ### the point on the surface
        #    print(p,'p')
        #   print(p)
        # print(i)
        vec = Poly_Approximated1_Normals["Normals"][i]  ### Normal Vector
        #   print(vec,'vec')

        p0 = p - 10000000 * vec
        p1 = p + 10000000 * vec

        ip, ic = Poly_Original1.ray_trace(p0, p1, first_point=True)  ######## ip = Contact point on the another surface
        #    print(ip, 'ip')

        dist = np.sqrt(np.sum((
                                      ip - p) ** 2))  #### if we consider p as a local origin, ip -p = vector of the distance. Also, we have a normal vector which is vec
        ### now if the inner product >0 it means ip and vec or in the same direction, otherwise, they are in the opoosit.
        # inner_product= (ip-p)[0]*vec[0] + (ip-p)[1]*vec[1] +(ip-p)[2]*vec[2]
        #   print(ip - p, "Delta")
        # print(Poly_Approximated1_Normals["Normals"][i])
        if len(ip) == 0:
            print("No touch", p)
            print(vec)
            Poly_Approximated1_Normals["distances"][i] = 0
            Contact_position_store.append(p)

        else:

            print("touch", p)
            inner_product = (ip - p)[0] * vec[0] + (ip - p)[1] * vec[1] + (ip - p)[2] * vec[2]
            if inner_product >= 0:
                Poly_Approximated1_Normals["distances"][i] = dist
                Contact_position_store.append(ip)
            else:
                Poly_Approximated1_Normals["distances"][i] = -1 * dist
                Contact_position_store.append(ip)
    # print(ip,'ip')
    # print(p,'p')
    # print(ip-p,"Delta")

    #  print(Poly_Approximated1_Normals["distances"][i],'dist')
    # Cost=0
    #  for j in range(0, d.shape[0], 1):
    #      Cost= Cost+ d[j]*d[j]

    return Poly_Approximated1_Normals["distances"], Contact_position_store


#################################################################################
def Generate_PolyDataMesh(mesh):
    add3 = []  # for surf.faces
    for e in np.array(np.array(mesh.faces)):
        ee = np.array(e)
        l = np.append(3, ee)
        add3.append(l)

    surf = pv.PolyData()
    surf.points = np.array(mesh.vertices, dtype=float)
    surf.faces = np.array(add3)
    return surf


###################################################################################
def Generate_PySubdivMesh(mesh):
    face3 = []
    for i in range(0, 4 * mesh.n_faces, 4):
        ee = np.array([mesh.faces[i + 1], mesh.faces[i + 2], mesh.faces[i + 3]])
        face3.append(ee)

    mesh2 = main.Mesh(mesh.points, face3)
    return mesh2


###################################################################################
def remeshing(mesh, n_devide, n_points):
    oo = Generate_PolyDataMesh(mesh)
    clus = pyacvd.Clustering(oo)
    clus.subdivide(n_devide)  ## 1 is the best in our work
    clus.cluster(
        n_points)  ## this number is very important.. 700 for approximated mesh  is nice or or 100 for control cage
    # remesh
    remesh = clus.create_mesh()
    mesh_remeshed = Generate_PySubdivMesh(remesh)
    return (mesh_remeshed)


def signed_distance_nearest_point(Original_mesh, Approximated_mesh):
    Poly_Original1 = Generate_PolyDataMesh(Original_mesh)
    Poly_Approximated1 = Generate_PolyDataMesh(Approximated_mesh)

    Poly_Approximated1_Normals = Poly_Approximated1.compute_normals(point_normals=True, cell_normals=False,
                                                                    flip_normals=True,
                                                                    auto_orient_normals=True)
    #  print(Poly_Approximated1_Normals.points)
    # Poly_Approximated1_Normals["distances"] = np.empty(Poly_Approximated1.n_points)
    tree = KDTree(Poly_Original1.points)
    d, idx = tree.query(Poly_Approximated1.points, k=1)
    # print(Poly_Original1.points[1])
    # print(Poly_Approximated1.points[1])
    # print(d[1])

    #  print(len(d),'d')
    # print(len(idx),'idx') ## idx = indices of the nearest vertex on the original mesh which is close to the approximated mesh
    #  d, idx = tree.query(Poly_Approximated1.points)

    # np.mean(d)
    Distance = np.empty(Poly_Approximated1.n_points)
    Contact_position_store = []

    for i in range(0, len(Poly_Approximated1.points)):

        indice = idx[i]
        # print(idx[i],'kirrrrrrrrrrrrr')

        p = Poly_Approximated1.points[i]  ### the point on the surface
        #    print(p,'p')
        #   print(p)
        # print(i)
        vec = Poly_Approximated1_Normals["Normals"][i]  ### Normal Vector
        #   print(vec,'vec')

        #  p0 = p - 10000000*vec
        #  p1 = p + 10000000*vec

        # ip, ic = Poly_Original1.ray_trace(p0, p1, first_point=True)######## ip = Contact point on the another surface

        ip = Poly_Original1.points[indice]
        #        Poly_Approximated1["distances"]= d
        #    print(ip, 'ip')
        # ip=Poly_Original1.points[idx][i]
        # dist = np.sqrt(np.sum((ip - p) ** 2))  #### if we consider p as a local origin, ip -p = vector of the distance. Also, we have a normal vector which is vec
        ### now if the inner product >0 it means ip and vec or in the same direction, otherwise, they are in the opoosit.
        # inner_product= (ip-p)[0]*vec[0] + (ip-p)[1]*vec[1] +(ip-p)[2]*vec[2]
        #   print(ip - p, "Delta")
        # print(Poly_Approximated1_Normals["Normals"][i])
        if len(idx) == 0:
            #  print ("No touch",p)
            #   print(vec)
            d[i] = 0
            Contact_position_store.append(p)

        else:

            # print("touch",p)
            #  print(ip)
            #   print(vec)
            inner_product = (ip - p)[0] * vec[0] + (ip - p)[1] * vec[1] + (ip - p)[2] * vec[2]
            if inner_product >= 0:
                d[i] = d[i]
                Contact_position_store.append(ip)
            else:
                d[i] = -1 * d[i]
                Contact_position_store.append(ip)
    # print(ip,'ip')
    # print(p,'p')
    # print(ip-p,"Delta")

    #  print(Poly_Approximated1_Normals["distances"][i],'dist')
    # Cost=0
    #  for j in range(0, d.shape[0], 1):
    #      Cost= Cost+ d[j]*d[j]

    return np.array(d), np.array(Contact_position_store)


def intersection_point(mesh, id1,
                       id2):  ### intersection point between one mesh and two points on top and down of the mesh

    Poly_mesh = Generate_PolyDataMesh(mesh)

    p0 = mesh.vertices[id1]
    p1 = mesh.vertices[id2]

    ip, ic = Poly_mesh.ray_trace(p0, p1, first_point=True)  ######## ip = Contact point on the another surface

    return ip
