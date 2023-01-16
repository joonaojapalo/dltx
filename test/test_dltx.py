import unittest

import numpy as np

from dltx import dlt_calibrate, dlt_reconstruct

DEBUG = False


def debug(msg):
    if not DEBUG:
        return
    print(msg)

class TestDLT (unittest.TestCase):
    def test_3d_4cams(self):
        # Tests of DLT
        debug('\nTEST #1')
        debug('Test of camera calibration and point reconstruction based on'
              ' direct linear transformation (DLT).')
        debug('3D (x, y, z) coordinates (in cm) of the corner of a cube:')
        xyz = [[0,    0,    0],
               [0, 12.3,    0],
               [14.5, 12.3,    0],
               [14.5,    0,    0],
               [0,    0, 14.5],
               [0, 12.3, 14.5],
               [14.5, 12.3, 14.5],
               [14.5,    0, 14.5]]
        debug(np.asarray(xyz))
        debug('2D (u, v) coordinates (in pixels) of 4 different views of the cube:')
        uv1 = [[1302, 1147],
               [1110,  976],
               [1411,  863],
               [1618, 1012],
               [1324,  812],
               [1127,  658],
               [1433,  564],
               [1645,  704]]
        uv2 = [[1094, 1187],
               [1130,  956],
               [1514,  968],
               [1532, 1187],
               [1076,  854],
               [1109,  647],
               [1514,  659],
               [1523,  860]]
        uv3 = [[1073,  866],
               [1319,  761],
               [1580,  896],
               [1352, 1016],
               [1064,  545],
               [1304,  449],
               [1568,  557],
               [1313,  668]]
        uv4 = [[1205, 1511],
               [1193, 1142],
               [1601, 1121],
               [1631, 1487],
               [1157, 1550],
               [1139, 1124],
               [1628, 1100],
               [1661, 1520]]
        debug('uv1:')
        debug(np.asarray(uv1))
        debug('uv2:')
        debug(np.asarray(uv2))
        debug('uv3:')
        debug(np.asarray(uv3))
        debug('uv4:')
        debug(np.asarray(uv4))

        debug('\nUse 4 views to perform a 3D calibration of the camera with 8 points of the cube:')
        n_dims = 3
        n_cams = 4
        L1, err1 = dlt_calibrate(n_dims, xyz, uv1)
        debug('Camera calibration parameters based on view #1:')
        debug(L1)
        debug('Error of the calibration of view #1 (in pixels):')
        debug(err1)
        self.assertLess(err1, 3.0, 'Error of the calibration of view #3 (in '
                        'pixels): not within tolerance')
        L2, err2 = dlt_calibrate(n_dims, xyz, uv2)
        debug('Camera calibration parameters based on view #2:')
        debug(L2)
        debug('Error of the calibration of view #2 (in pixels):')
        debug(err2)
        self.assertLess(err2, 3.1,
                        'Error of the calibration of view #3 (in '
                        'pixels): not within tolerance')
        L3, err3 = dlt_calibrate(n_dims, xyz, uv3)
        debug('Camera calibration parameters based on view #3:')
        debug(L3)
        debug('Error of the calibration of view #3 (in pixels):')
        debug(err3)
        self.assertLess(err3, 6.5,
                        'Error of the calibration of view #3 (in'
                        'pixels): not within tolerance')
        L4, err4 = dlt_calibrate(n_dims, xyz, uv4)
        debug('Camera calibration parameters based on view #4:')
        debug(L4)
        debug('Error of the calibration of view #4 (in pixels):')
        debug(err4)
        self.assertLess(err4, 3.0,
                        'Error of the calibration of view #4 (in '
                        'pixels): not within tolerance')
        xyz1234 = np.zeros((len(xyz), 3))
        L1234 = [L1, L2, L3, L4]
        for i in range(len(uv1)):
            xyz1234[i, :] = dlt_reconstruct(n_dims, n_cams, L1234, [
                                            uv1[i], uv2[i], uv3[i], uv4[i]])
        debug('Reconstruction of the same 8 points based on 4 views and the camera calibration parameters:')
        debug(xyz1234)
        debug('Mean error of the point reconstruction using the DLT (error in cm):')
        err = np.mean(
            np.sqrt(np.sum((np.array(xyz1234) - np.array(xyz))**2, 1)))
        debug(err)
        self.assertLess(err, 0.11,
                        'Mean error of the point reconstruction '
                        'using the DLT (error in cm) is not within tolerance')

    def test_2d(self):
        debug('\nTEST #2')
        debug('Test of the 2D DLT')
        debug('2D (x, y) coordinates (in cm) of the corner of a square:')
        xy = [[0,    0],
              [0, 12.3],
              [14.5, 12.3],
              [14.5,    0]]
        debug(np.asarray(xy))
        debug('2D (u, v) coordinates (in pixels) of 2 different views of the square:')
        uv1 = [[1302, 1147],
               [1110,  976],
               [1411,  863],
               [1618, 1012]]
        uv2 = [[1094, 1187],
               [1130,  956],
               [1514,  968],
               [1532, 1187]]
        debug('uv1:')
        debug(np.asarray(uv1))
        debug('uv2:')
        debug(np.asarray(uv2))
        debug('')
        debug('Use 2 views to perform a 2D calibration of the camera with 4 points of the square:')
        n_dims = 2
        n_cams = 2
        L1, err1 = dlt_calibrate(n_dims, xy, uv1)
        debug('Camera calibration parameters based on view #1:')
        debug(L1)
        debug('Error of the calibration of view #1 (in pixels):')
        debug(err1)
        self.assertLess(err1, 1e-9,
                        'Error of the calibration of view #1 (in '
                        'pixels) is not within tolerance')

        L2, err2 = dlt_calibrate(n_dims, xy, uv2)
        debug('Camera calibration parameters based on view #2:')
        debug(L2)
        debug('Error of the calibration of view #2 (in pixels):')
        debug(err2)
        self.assertLess(err2, 1e-9,
                        'Error of the calibration of view #2 (in '
                        'pixels) is not within tolerance')

        xy12 = np.zeros((len(xy), 2))
        L12 = [L1, L2]
        for i in range(len(uv1)):
            xy12[i, :] = dlt_reconstruct(n_dims, n_cams, L12, [uv1[i], uv2[i]])
        debug('Reconstruction of the same 4 points based on 2 views and the camera calibration parameters:')
        debug(xy12)
        debug('Mean error of the point reconstruction using the DLT (error in cm):')
        err = np.mean(np.sqrt(np.sum((np.array(xy12) - np.array(xy))**2, 1)))
        debug(err)
        self.assertLess(err, 1e-10, 'Mean error of the point reconstruction '
                        'using the DLT (error in cm) is not within tolerance')

    def test_2d_1cam(self):
        debug('\nTEST #3')
        debug('Use only one view to perform a 2D calibration of the camera'
              ' with 4 points of the square:')
        n_dims = 2
        n_cams = 1
        xy = [[0,    0],
              [0, 12.3],
              [14.5, 12.3],
              [14.5,    0]]
        uv1 = [[1302, 1147],
               [1110,  976],
               [1411,  863],
               [1618, 1012]]
        L1, err1 = dlt_calibrate(n_dims, xy, uv1)
        debug('Camera calibration parameters based on view #1:')
        debug(L1)
        debug('Error of the calibration of view #1 (in pixels):')
        debug(err1)
        self.assertLess(
            err1, 1e-9,
            "Error of the calibration of view  # 1 (in pixels) is "
            "not within tolerance.")
        xy1 = np.zeros((len(xy), 2))

        for i in range(len(uv1)):
            xy1[i, :] = dlt_reconstruct(n_dims, n_cams, L1, uv1[i])
        debug('Reconstruction of the same 4 points based on one view and'
              ' the camera calibration parameters:')
        debug(xy1)

        debug('Mean error of the point reconstruction using the DLT (error in cm):')
        err = np.mean(np.sqrt(np.sum((np.array(xy1) - np.array(xy))**2, 1)))
        debug(err)
        self.assertLess(err, 1e-1)


if __name__ == '__main__':
    unittest.main()
