def set_crease_interactive(self):
    """
    Set the crease for the edges which are in the unique_edges array by picking vertices forming an edge in an
    interactive PyVista widget.
    When called the first time creates np.array with zeros of length unique_edges.

    Returns
    --------
        edges_crease: (n,1) float
            np.array with creases for corresponding edge in the unique_edges array
    """

    crease_arr = visualize.set_creases_visualize(self)
    self.data['creases'] = crease_arr
    self.set_crease()
    return self.data['creases']

def recalculate_face_normals(self):
        """
        Trying to recalibrate face normals by sending out rays. If ray hits a points on another face, the normal is
        pointing inside. The face normal is flagged to be flipped.

        Parameters
        --------------
        self : PySubdiv mesh object
            mesh on which the face normals should be calculated
        Returns
        -------
        face_normals : (n,3)
        """

        # array to store the flag state of the normal which will be multiplied with the face normal array;
        # 1 stay -1 flip
        normals_to_flip = np.zeros(len(self.faces))
        # loop over each face and cast a ray for each face from the centroids position in normal direction
        for face_idx in range(len(self.faces)):
            # return of the function, often also start of ray cast is registered as hit
            points = calculation.ray_cast(self.face_centroids[face_idx], self.face_normals[face_idx], self.model())[0]
            # if two or more points are hit flag to flip normal; probably first hit is start point second hit should be
            # on the opposite face
            if len(points) >= 2:
                normals_to_flip[face_idx] = -1
            else:
                # if zero or one points is reverse start and end point of the ray
                points_reversed = calculation.ray_cast(self.face_centroids[face_idx], self.face_normals[face_idx],
                                                       self.model(), back_cast=True)[0]
                # if no point is hit on the first ray cast, face normal is probably oriented outwards, flag normal as
                # correct
                if len(points) == 0 or len(points_reversed) == 0:
                    normals_to_flip[face_idx] = 1
                else:
                    # if length of point hit array is one we will check the hit point against the startpoint of the ray
                    points_tuple = tuple(points[0])
                    points_reversed_tuple = tuple(points_reversed[0])
                    face_centroid_tuple = tuple(self.face_centroids[face_idx])
                    boolean_mask = []
                    # loop over the x, y, z coordinate of the hit point and check against the start point
                    # if coordinates are equal continue otherwise flag normals to be flipped
                    for i, coordinate in enumerate(points_tuple):
                        if np.allclose(coordinate, face_centroid_tuple[i]):
                            boolean_mask.append(True)
                        else:
                            boolean_mask.append(False)
                    if all(boolean_mask):
                        # start is equal to hit -> check if start is equal to hit of reverse ray cast if not equal
                        # flag to flip the normal
                        if {points_tuple} != {points_reversed_tuple}:
                            normals_to_flip[face_idx] = -1
                        else:
                            # first hit is equal to reverse hit-> face normal probably pointing outwards
                            # no flip necessary
                            normals_to_flip[face_idx] = 1
                    else:
                        # start is not equal to hit -> flip the normal
                        normals_to_flip[face_idx] = -1
        self.face_normals = self.face_normals * normals_to_flip.reshape((-1, 1))
        self.define_vertex_normals()
        self.data['face_normals'] = self.face_normals
        return self.face_normals
