# DLT

This package implements camera calibration and point reconstruction by *direct linear
transformation* (DLT).

## Example

```python
from dltx import dlt_calibrate, dlt_reconstruct

# Define locations of 6 or more points in real world.
world_positions = [
    [0,     0,      2550],  # point 1
    [0,     0,      0],     # point 2
    [0,     2632,   0   ],  # point 3
    [4500,  0,      2550],  # point 4
    [5000,  0,      0   ],  # point 5
    [5660,  2620,   0   ]   # point 6
]

# Define pixel coordinates of respective points seen by two or 
# more cameras.
cameras = [
    # Camera 1
    [
        [1810, 885],
        [1353, 786],
        [1362, 301],
        [455, 1010],
        [329, 832],
        [183, 180]
    ],
    # Camera 2
    [
        [1734, 952],
        [1528, 768],
        [1546, 135],
        [115, 834],
        [459, 719],
        [358, 202]
    ]
]

# Calibrate cameras
n_dims = 3
L1, err = dlt_calibrate(n_dims, world_positions, cameras[0])
L2, err = dlt_calibrate(n_dims, world_positions, cameras[1])
camera_calibration = [L1, L2]

# Find world coordinates for `query_point` visible in both cameras
query_point = [
    [1810, 885], # cam 1
    [1734, 952]  # cam 2
    ]
dlt_reconstruct(n_dims, len(cameras), camera_calibration, query_point)
# coordinates in real world: [-1.31704156e-01,  8.71539661e-01,  2.54975288e+03]
```

## Background

The fundamental problem here is to find a mathematical relationship between the
coordinates of a 3D point and its projection onto the image plane. The DLT
(a linear approximation to this problem) is derived from modeling the object
and its projection on the image plane as a pinhole camera situation.

In simplistic terms, using a pinhole camera model, it can be found by similar
triangles the following relation between the image coordinates (u,v) and the 3D
point (X,Y,Z):

```text
    [ u ]   [ L1  L2  L3  L4 ]   [ X ]
    [ v ] = [ L5  L6  L7  L8 ] * [ Y ]
    [ 1 ]   [ L9 L10 L11 L12 ]   [ Z ]
                                 [ 1 ]
```

The matrix L is kwnown as the camera matrix or camera projection matrix. For a
2D point (X,Y), this matrix is 3x3. In fact, the L12 term (or L9 for 2D DLT)
is not independent from the other parameters and then there are only 11
(or 8 for 2D DLT) independent parameters in the DLT to be determined through
the calibration procedure.

There are more accurate (but more complex) algorithms for camera calibration
that also consider lens distortion. For example, OpenCV and Tsai softwares have
been ported to Python. However, DLT is classic, simple, and effective (fast)
for most applications.

About DLT, see: https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html

## Usage

Methods for camera calibration and point reconstruction based on DLT.
DLT is typically used in two steps:

**1. Camera calibration:**

```python
L, err = dlt_calibrate(nd, xyz, uv)
```

**2. Object (point) reconstruction**

```python
xyz = dlt_reconstruct(n_dims, n_cams, Ls, uvs)
```

The camera calibration step consists in digitizing points with known
coordinates in the real space and find the camera parameters.

At least 4 points are necessary for the calibration of a plane (2D DLT)
and at least 6 points for the calibration of a volume (3D DLT). For the 2D DLT,
at least one view of the object (points) must be entered. For the 3D DLT, at
least 2 different views of the object (points) must be entered.

These coordinates (from the object and image(s)) are inputed to the
`dlt_calibrate` algorithm which estimates the camera parameters (8 for 2D DLT
and 11 for 3D DLT).

Usually it is used more points than the minimum necessary and the
overdetermined linear system is solved by a least squares minimization
algorithm. Here this problem is solved using singular value
decomposition (SVD).

With these camera parameters and with the camera(s) at the same position
of the calibration step, we now can reconstruct the real position of any
point inside the calibrated space (area for 2D DLT and volume for the
3D DLT) from the point position(s) viewed by the same fixed camera(s).

This code can perform 2D or 3D DLT with any number of views (cameras).
For 3D DLT, at least two views (cameras) are necessary.

## Testing

Run unit test suite:

```sh
python -m unittest test.test_dltx
```
