#!/usr/bin/env python

__author__ = "Marcos Duarte <duartexyz@gmail.com>"
__version__ = "DLT.py v.0.0.2 2023/01/13"


import numpy as np


def dlt_calibrate(n_dims, xyz, uv):
    """
    Camera calibration by DLT with known object points and image points
    This code performs 2D or 3D DLT camera calibration with any number of
        views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    Inputs:
        n_dims is the number of dimensions of the object space: 3 for 3D DLT
        and 2 for 2D DLT.
        xyz are the coordinates in the object 3D or 2D space of the
        calibration points.
        uv are the coordinates in the image 2D space of these calibration
        points.
        The coordinates (x,y,z and u,v) are given as columns and the different
        points as rows.
        For the 2D DLT (object planar space), only the first 2 columns
        (x and y) are used.
        There must be at least 6 calibration points for the 3D DLT and 4
        for the 2D DLT.
    Outputs:
        L: array of the 8 or 11 parameters of the calibration matrix.
        err: error of the DLT (mean residual of the DLT transformation in units 
        of camera coordinates).
    """

    # Convert all variables to numpy array:
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)
    # Number of points:
    n_points = xyz.shape[0]
    # Check the parameters:
    if uv.shape[0] != n_points:
        raise ValueError('xyz (%d points) and uv (%d points) have different number of points.'
                            % (n_points, uv.shape[0]))
    if (n_dims == 2 and xyz.shape[1] != 2) or (n_dims == 3 and xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).'
                            % (xyz.shape[1], n_dims, n_dims))
    if n_dims == 3 and n_points < 6 or n_dims == 2 and n_points < 4:
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.'
                            % (n_dims, 2*n_dims, n_points))

    # Normalize the data to improve the DLT quality (DLT is dependent on the
    #  system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1
    #  at each direction.
    Txyz, xyzn = normalize(n_dims, xyz)
    Tuv, uvn = normalize(2, uv)
    # Formulating the problem as a set of homogeneous linear equations, M*p=0:
    A = []
    if n_dims == 2:  # 2D DLT
        for i in range(n_points):
            x, y = xyzn[i, 0], xyzn[i, 1]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    elif n_dims == 3:  # 3D DLT
        for i in range(n_points):
            x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v])

    # Convert A to array:
    A = np.asarray(A)
    # Find the 11 (or 8 for 2D DLT) parameters:
    U, S, Vh = np.linalg.svd(A)
    # The parameters are in the last line of Vh and normalize them:
    L = Vh[-1, :] / Vh[-1, -1]
    # Camera projection matrix:
    H = L.reshape(3, n_dims+1)
    # Denormalization:
    H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz)
    H = H / H[-1, -1]
    L = H.flatten()
    # Mean error of the DLT (mean residual of the DLT transformation in
    #  units of camera coordinates):
    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]
    # Mean distance:
    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv)**2, 1)))

    return L, err

def dlt_reconstruct(n_dims, n_cams, Ls, uvs):
    """
    Reconstruction of object point from image point(s) based on the DLT parameters.
    This code performs 2D or 3D DLT point reconstruction with any number of
        views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    Inputs:
        n_dims is the number of dimensions of the object space: 3 for 3D DLT
        and 2 for 2D DLT.
        n_cams is the number of cameras (views) used.
        Ls (array type) are the camera calibration parameters of each camera 
        (is the output of DLTcalib function). The Ls parameters are given
        as columns and the Ls for different cameras as rows.
        uvs are the coordinates of the point in the image 2D space of each camera.
        The coordinates of the point are given as columns and the different
        views as rows.
    Outputs:
        xyz: point coordinates in space.
    """

    # Convert Ls to array:
    Ls = np.asarray(Ls)
    # Check the parameters:
    if Ls.ndim == 1 and n_cams != 1:
        raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (1) are different.'
                            % (n_cams))
    if Ls.ndim > 1 and n_cams != Ls.shape[0]:
        raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (%d) are different.'
                            % (n_cams, Ls.shape[0]))
    if n_dims == 3 and Ls.ndim == 1:
        raise ValueError(
            'At least two sets of camera calibration parametersd are neede for 3D point reconstruction.')

    # 2D and 1 camera (view), the simplest (and fastest) case.
    if n_cams == 1:
        # One could calculate inv(H) and input that to the code to speed up
        #  things if needed.
        Hinv = np.linalg.inv(Ls.reshape(3, 3))
        # Point coordinates in space:
        xyz = np.dot(Hinv, [uvs[0], uvs[1], 1])
        xyz = xyz[0:2] / xyz[2]
    else:
        # Formulate problem as a set of homogeneous linear equations, A*p=0:
        M = []
        for i in range(n_cams):
            L = Ls[i, :]
            # indexing works for both list and numpy array
            u, v = uvs[i][0], uvs[i][1]
            if n_dims == 2:
                M.append([L[0]-u*L[6], L[1]-u*L[7], L[2]-u*L[8]])
                M.append([L[3]-v*L[6], L[4]-v*L[7], L[5]-v*L[8]])
            elif n_dims == 3:
                M.append([L[0]-u*L[8], L[1]-u*L[9],
                            L[2]-u*L[10], L[3]-u*L[11]])
                M.append([L[4]-v*L[8], L[5]-v*L[9],
                            L[6]-v*L[10], L[7]-v*L[11]])

        # Find the xyz coordinates:
        U, S, Vh = np.linalg.svd(np.asarray(M))
        # Point coordinates in space:
        xyz = Vh[-1, 0:-1] / Vh[-1, -1]

    return xyz

def normalize(n_dims, x):
    """Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3)).
    Inputs:
        n_dims: number of dimensions (2 for 2D; 3 for 3D)
        x: the data to be normalized (directions at different columns and points at rows)
    Outputs:
        Tr: the transformation matrix (translation plus scaling)
        x: the transformed data
    """

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if n_dims == 2:
        Tr = np.array([[s, 0, m[0]],
                        [0, s, m[1]],
                        [0, 0,   1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]],
                        [0, s, 0, m[1]],
                        [0, 0, s, m[2]],
                        [0, 0, 0,   1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:n_dims, :].T

    return Tr, x

