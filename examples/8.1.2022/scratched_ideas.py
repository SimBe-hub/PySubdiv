def recalculate_face_normals(self):
    print(self.data['edges'])
    print(self.faces)
    faces_edges = self.data['edges'].reshape(-1, 3, 2)

    # create list of sets -> each set is a face (index of set in list) with tuples of the edges
    faces_list = []
    for face in faces_edges:
        edge_set = set()
        for edge in face:
            edge_set.add(tuple(edge))
        faces_list.append(edge_set)

    flags = np.zeros(len(self.faces))
    # loop over each face and check if edges of that face are in other faces as well -> directed edges
    for face_index, face in enumerate(faces_list):
        for edge in face:
            for check_face_index, check_face in enumerate(
                    faces_list[:face_index] + [{(0, 0, 0), (0, 0, 0), (0, 0, 0)}] + faces_list[face_index + 1:]):
                if edge in check_face:
                    print(edge)
                    print(face_index)
                    print(check_face_index)
                    print(check_face)
                    flags[face_index] += 1
                    flags[check_face_index] += 1
    print(flags)
    normals_to_flip = np.where(flags >= 5, -1, 1)
    print(normals_to_flip.reshape(-1, 1))
    print(self.face_normals)
    self.face_normals = np.multiply(self.face_normals, normals_to_flip.reshape(-1, 1))
    print(self.face_normals)
    self.data['face_normals'] = self.face_normals
    return self.face_normals