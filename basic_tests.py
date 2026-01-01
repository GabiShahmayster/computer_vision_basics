import time
import unittest
import numpy as np
from src.CameraModels import PinholeCamera
from src.CameraPose import CameraIntrinsics, CameraPose
from scipy.spatial.transform import Rotation
import cv2

from src.Transformations import HomogeneousTransformation3D
from src.VideoTools import ColorTuple


def check_epipolar_constraint(frame_1_coordinates: np.ndarray,
                              frame_2_coordinates: np.ndarray,
                              epipolar_matrix: np.ndarray) -> float:
    frame_1_coordinates = frame_1_coordinates.squeeze()
    frame_2_coordinates = frame_2_coordinates.squeeze()
    return abs(np.array([frame_1_coordinates[0], frame_1_coordinates[1], 1.0]).T \
               @ epipolar_matrix \
               @ np.array([frame_2_coordinates[0], frame_2_coordinates[1], 1.0]))


def get_normalization_matrix(pixel_coordinates: np.ndarray) -> np.ndarray:
    """
    normalize pixel coordinates to have an average of 0 and a std.dev of 1
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
    """
    out: np.ndarray = pixel_coordinates.__copy__()

    # centralize points
    T: np.ndarray = np.eye(3)
    T[0, 0] = 2 / (np.var(out[:, 0]) ** 1)
    T[0, 2] = -np.mean(out[:, 0]) * T[0, 0]
    T[1, 1] = 2 / (np.var(out[:, 1]) ** 1)
    T[1, 2] = -np.mean(out[:, 1]) * T[1, 1]

    return T


def get_projection_matrix(camera_translation: np.ndarray,
                          camera_rotation: np.ndarray,
                          image_width: int,
                          image_height: int,
                          focal_length_pixel: int) -> np.ndarray:
    # camera model
    K = get_camera_matrix(image_width=image_width,
                          image_height=image_height,
                          focal_length_pixel=focal_length_pixel)

    # camera pose (homogeneous)
    # C = [R | -R*t]
    C = np.hstack((camera_rotation, -camera_rotation @ camera_translation))

    # projection matrix
    return K @ C

def get_camera_matrix(image_width: int,
                      image_height: int,
                      focal_length_pixel: int) -> np.ndarray:
    return np.array([[focal_length_pixel, .0, image_width / 2],
                     [.0, focal_length_pixel, image_height / 2],
                     [.0, .0, 1.0]])

class unitTests(unittest.TestCase):

    def test_fundamental_matrix_estimation(self):
        """
        This tests uses the projections of 8 cube vertices to 2 camera poses, to estimate the fundamental matrix
        def compute_fundamental(x1,x2):
        """

        N_POINTS: int = 8

        cube_world_points: np.ndarray = np.array([[0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
                                                  [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                                  [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]])

        FRAME_WIDTH: float = 800.0
        FRAME_HEIGHT: float = 700.0
        FOCAL_LENGTH_PIXEL: float = 1000.0
        camera_matrix: np.ndarray = np.array([[FOCAL_LENGTH_PIXEL, .0, FRAME_WIDTH / 2],
                                              [.0, FOCAL_LENGTH_PIXEL, FRAME_HEIGHT / 2],
                                              [.0, .0, 1.0]])
        frame_1: np.ndarray = np.zeros((int(FRAME_HEIGHT), int(FRAME_WIDTH)))
        frame_2: np.ndarray = np.zeros((int(FRAME_HEIGHT), int(FRAME_WIDTH)))

        DELTA_TRANSLATION: np.ndarray = np.array([1.0,
                                                  .0,
                                                  1.0])
        DELTA_HEADING_DEG: float = .0
        DELTA_PITCH_DEG: float = 10.0
        DELTA_ROLL_DEG: float = .0

        intrinsics: CameraIntrinsics = CameraIntrinsics(camera_matrix=camera_matrix,
                                                        camera_distortion=None,
                                                        frame_width_px=FRAME_WIDTH,
                                                        frame_height_px=FRAME_HEIGHT)
        camera_pose_frame_1: CameraPose = CameraPose(rotation=Rotation.from_matrix(np.eye(3)),
                                                     translation=np.array([.0, .0, -10.0]))
        camera_frame_1: PinholeCamera = PinholeCamera(parameters=intrinsics,
                                                      camera_pose=camera_pose_frame_1)

        projected_points_frame_1, _ = camera_frame_1.return_valid_image_points(points=cube_world_points)
        for pt_idx in range(N_POINTS):
            cv2.drawMarker(frame_1,
                       (int(projected_points_frame_1[pt_idx, 0]), int(projected_points_frame_1[pt_idx, 1])),
                       ColorTuple.white.value)
        cv2.imshow("frame 1", frame_1)

        camera_1_yaw_deg, camera_1_pitch_deg, camera_1_roll_deg = camera_pose_frame_1.rotation.as_euler('ZYX', degrees=True)
        camera_rotation_frame_2: Rotation = Rotation.from_euler('ZYX', [camera_1_yaw_deg + DELTA_HEADING_DEG,
                                                                        camera_1_pitch_deg + DELTA_PITCH_DEG,
                                                                        camera_1_roll_deg + DELTA_ROLL_DEG],
                                                                degrees=True)
        camera_pose_frame_2: CameraPose = CameraPose(rotation=camera_rotation_frame_2,
                                                     translation=camera_pose_frame_1.translation + DELTA_TRANSLATION)
        camera_frame_2: PinholeCamera = PinholeCamera(parameters=intrinsics,
                                                      camera_pose=camera_pose_frame_2)

        projected_points_frame_2, _ = camera_frame_2.return_valid_image_points(points=cube_world_points)
        for pt_idx in range(N_POINTS):
            cv2.drawMarker(frame_2,
                       (int(projected_points_frame_2[pt_idx, 0]), int(projected_points_frame_2[pt_idx, 1])),
                       ColorTuple.white.value)
        cv2.imshow("frame 2", frame_2)

        # step 1 - pre-scaling
        T_1: np.ndarray = get_normalization_matrix(projected_points_frame_1)
        T_2: np.ndarray = get_normalization_matrix(projected_points_frame_2)

        normalized_points_1 = T_1 @ np.vstack((projected_points_frame_1.T, np.ones((1, N_POINTS))))
        normalized_points_2 = T_2 @ np.vstack((projected_points_frame_2.T, np.ones((1, N_POINTS))))
        # step 2 - build the "equation matrix", A
        A: np.ndarray = np.zeros((N_POINTS, 9))
        for pt_idx in range(N_POINTS):
            u = normalized_points_1.T[pt_idx, 0]
            v = normalized_points_1.T[pt_idx, 1]
            u_tag = normalized_points_2.T[pt_idx, 0]
            v_tag = normalized_points_2.T[pt_idx, 1]
            A[pt_idx, :] = np.array([u * u_tag,
                                     u * v_tag,
                                     u,
                                     v * u_tag,
                                     v * v_tag,
                                     v,
                                     u_tag,
                                     v_tag,
                                     1.0])

        # step 3 - find solution to A*f=0, where f is a vector of the fundamental matrix F components, using SVD
        U, S, V = np.linalg.svd(A)
        F = V[:, -1].reshape(3, 3)
        U, S, V = np.linalg.svd(F)
        S[2] = 0  # force rank = 2
        fundamental_mat_normalized = U @ np.diag(S) @ V
        fundamental_mat = T_2.T @ fundamental_mat_normalized @ T_1

        fundamental_mat_OpenCV, _ = cv2.findFundamentalMat(points1=projected_points_frame_1,
                                                       points2=projected_points_frame_2,
                                                       method=cv2.FM_8POINT)

        DEFAULT_NISTER_PROB: float = .999
        DEFAULT_NISTER_THRESHOLD: float = 1.0
        essential_mat_OpenCV, _ = cv2.findEssentialMat(points1=projected_points_frame_1,
                                                   points2=projected_points_frame_2,
                                                   cameraMatrix=camera_matrix,
                                                   method=cv2.RANSAC,
                                                   prob=DEFAULT_NISTER_PROB,
                                                   threshold=DEFAULT_NISTER_THRESHOLD)

        # check epipolar constraint for essential matrix
        # SAD = sum of absolute differences
        SAD_essential_OpenCV = .0
        points_frame_1_homogeneous: np.ndarray = np.vstack((projected_points_frame_1.T, np.ones((1, N_POINTS))))
        points_frame_2_homogeneous: np.ndarray = np.vstack((projected_points_frame_1.T, np.ones((1, N_POINTS))))
        K_inv = np.linalg.inv(camera_matrix)
        for n in range(N_POINTS):
            SAD_essential_OpenCV += check_epipolar_constraint(
                frame_1_coordinates=K_inv @ points_frame_1_homogeneous[:, n],
                frame_2_coordinates=K_inv @ points_frame_2_homogeneous[:, n],
                epipolar_matrix=essential_mat_OpenCV)
        print("OpenCV's essential matrix SAD = {0:f}".format(SAD_essential_OpenCV))

        # check epipolar constraint for directly computed fundamental matrix
        SAD_fundamental = .0
        for n in range(N_POINTS):
            SAD_fundamental += check_epipolar_constraint(frame_1_coordinates=projected_points_frame_1[n, :],
                                                         frame_2_coordinates=projected_points_frame_2[n, :],
                                                         epipolar_matrix=fundamental_mat)
        print("tested fundamental matrix SAD = {0:f}".format(SAD_fundamental))

        cv2.waitKey(0)
        if False:
            # check epipolar constraint for fundamental matrix
            # SAD = sum of absolute differences
            SAD_fundamental_OpenCV = .0
            for n in range(N_POINTS):
                SAD_fundamental_OpenCV += check_epipolar_constraint(frame_1_coordinates=projected_points_frame_1[n, :],
                                                                    frame_2_coordinates=projected_points_frame_2[n, :],
                                                                    epipolar_matrix=fundamental_mat_OpenCV)
            print("OpenCV's fundamental matrix SAD = {0:f}".format(SAD_fundamental_OpenCV))
            np.testing.assert_array_almost_equal(fundamental_mat_OpenCV, fundamental_mat)


    def test_PnP(self):
        """
        This tests estimates the pose of the camera, given the 3D world coordinates of 8 cube vertices and camera
        intrinsics
        :return:
        """
        N_point: int = 8

        cube_world_points: np.ndarray = np.array([[0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
                                                  [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                                  [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]]).astype(float).T

        cube_projection_result: np.ndarray = np.array([[511.11111111, 238.88888889],
                                                       [490.90909091, 259.09090909],
                                                       [511.11111111, 461.11111111],
                                                       [490.90909091, 440.90909091],
                                                       [288.88888889, 238.88888889],
                                                       [309.09090909, 259.09090909],
                                                       [288.88888889, 461.11111111],
                                                       [309.09090909, 440.90909091]]).astype(float).reshape(
            (N_point, 1, 2))

        reference_camera_translation: np.ndarray = np.array([.0,
                                                             .0,
                                                             -5.0])
        reference_camera_rotation: np.ndarray = np.eye(3)

        image_width: int = 800
        image_height: int = 700
        focal_length_pixel: int = 1000

        # camera model
        K = np.array([[focal_length_pixel, .0, image_width / 2],
                      [.0, focal_length_pixel, image_height / 2],
                      [.0, .0, 1.0]]).astype(float)

        (_, rotation_vector, translation_vector) = cv2.solvePnP(objectPoints=cube_world_points,
                                                            imagePoints=cube_projection_result,
                                                            cameraMatrix=K,
                                                            distCoeffs=None)
        rotation_matrix: np.ndarray = cv2.Rodrigues(rotation_vector)[0]

        np.testing.assert_array_almost_equal(-rotation_matrix @ translation_vector.squeeze(),
                                             reference_camera_translation)
        np.testing.assert_array_almost_equal(rotation_matrix, reference_camera_rotation)


    def test_open3d_cube(self):
        import open3d as o3d
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        FRAME_WIDTH: float = 800.0
        FRAME_HEIGHT: float = 700.0
        CUBE_POSITION: np.ndarray = np.array([.0, .0, 4.0])
        CAMERA_POSITION: np.ndarray = np.zeros(3)

        geom1 = o3d.geometry.TriangleMesh.create_box(width=.5, height=.5, depth=.5)
        geom1.translate((CUBE_POSITION[0], CUBE_POSITION[1], CUBE_POSITION[2]), relative=False)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(FRAME_WIDTH), height=int(FRAME_HEIGHT))

        vis.add_geometry(geom1)

        ctrl = vis.get_view_control()

        # while vis.poll_events():
        # rotate the camera
        camera_params = ctrl.convert_to_pinhole_camera_parameters()
        camera_params.intrinsic.set_intrinsics(width=int(FRAME_WIDTH),
                                               height=int(FRAME_HEIGHT),
                                               fx=1000.0,
                                               fy=1000.0,
                                               cx=(FRAME_WIDTH - 1) / 2,
                                               cy=(FRAME_HEIGHT - 1) / 2)

        camera_pose_homogeneous: HomogeneousTransformation3D = HomogeneousTransformation3D(
            rotation=Rotation.from_matrix(np.eye(3)),
            translation=CAMERA_POSITION)

        # TODO object should be transformed w.r.t camera, by calling translate(x, relative=False)
        camera_params.extrinsic = np.linalg.inv(camera_pose_homogeneous.as_se3())
        ctrl.convert_from_pinhole_camera_parameters(camera_params)
        # vis.poll_events()
        # vis.update_renderer()
        img: o3d.geometry.Image = vis.capture_screen_float_buffer(True)
        image_open3d = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        time.sleep(.1)
        # vis.destroy_window()

        # construct PinholeCameraObject
        intrinsics: CameraIntrinsics = CameraIntrinsics.from_open3d_intrinsics(camera_params.intrinsic)
        camera_pose: CameraPose = CameraPose(rotation=Rotation.from_matrix(np.eye(3)),
                                             translation=CAMERA_POSITION)
        blender_camera: PinholeCamera = PinholeCamera(parameters=intrinsics,
                                                      camera_pose=camera_pose)

        cube_vertices: np.ndarray = np.array(geom1.vertices)
        projected_points, _ = blender_camera.return_valid_image_points(points=cube_vertices.T)

        # draw projected points
        for pt in projected_points:
            cv2.drawMarker(image_open3d,
                       (int(pt[0]), int(pt[1])),
                       (0, 0, 0))

        # display OpenCV image
        cv2.imshow("Open3D render + theoretical projection", image_open3d)
        cv2.waitKey(0)


    def test_perspective_projection(self):
        """
        This tests project 8 cube vertices to the camera image plane, given the world coordinates of the vertices,
        camera extrinstic and intrinsic parameters
        :return:
        """
        N_POINTS = 8

        cube_world_points: np.ndarray = np.array([[0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
                                                  [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                                  [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]])

        cube_projection_result: np.ndarray = np.array([[511.11111111, 238.88888889],
                                                       [490.90909091, 259.09090909],
                                                       [511.11111111, 461.11111111],
                                                       [490.90909091, 440.90909091],
                                                       [288.88888889, 238.88888889],
                                                       [309.09090909, 259.09090909],
                                                       [288.88888889, 461.11111111],
                                                       [309.09090909, 440.90909091]])

        camera_translation: np.ndarray = np.array([.0,
                                                   .0,
                                                   -5.0]).reshape((3, 1))
        camera_rotation: np.ndarray = np.eye(3)

        image_width: int = 800
        image_height: int = 700
        focal_length_pixel: int = 1000

        # camera model
        K = np.array([[focal_length_pixel, .0, image_width / 2],
                      [.0, focal_length_pixel, image_height / 2],
                      [.0, .0, 1.0]])

        # camera pose (homogeneous)
        # C = [R | -R*t]
        C = np.hstack((camera_rotation, -camera_rotation @ camera_translation))

        # projection matrix
        P = K @ C

        # 3D world points (homogeneous)
        X_world_homogeneous = np.vstack((cube_world_points, np.ones((1, N_POINTS))))

        # transform to sensor-frame
        X_cam_homogeneous = P @ X_world_homogeneous

        # project to image
        cube_projection = np.empty((2, N_POINTS))

        for n in range(N_POINTS):
            cube_projection[:, n] = X_cam_homogeneous[:2, n] / X_cam_homogeneous[2, n]

        np.testing.assert_array_almost_equal(cube_projection, cube_projection_result.T)

        # out = np.zeros(image_height, image_width)
        # for p in cube_projection:
        #     a=2


    def test_projection_matrix_decomposition(self):

        """
        This tests project 8 cube vertices to the camera image plane, given the world coordinates of the vertices,
        camera extrinstic and intrinsic parameters
        :return:
        """
        image_width: int = 800
        image_height: int = 700
        focal_length_pixel: int = 1000
        K: np.ndarray = get_camera_matrix(image_width=image_width,
                                          image_height=image_height,
                                          focal_length_pixel=focal_length_pixel)
        camera_translation: np.ndarray = np.array([.0,
                                                   .0,
                                                   -5.0]).reshape((3, 1))
        camera_rotation: np.ndarray = np.eye(3)
        projection_matrix: np.ndarray = get_projection_matrix(camera_translation=camera_translation,
                                              camera_rotation=camera_rotation.T,
                                              image_width=image_width,
                                              image_height=image_height,
                                              focal_length_pixel=focal_length_pixel)
        # H=K*R
        H = projection_matrix[:3, :3]
        # h=-K*R*t=-H*t
        h = projection_matrix[:3, 3]

        camera_translation_est = -np.linalg.inv(H) @ h
        np.testing.assert_array_almost_equal(camera_translation_est.squeeze(), camera_translation.squeeze())


        # H^(-1)=R^(-1)*K^(-1)
        H_inv = np.linalg.inv(H)
        # A = Q * R, A- real square matrix, Q orthogonal, R upper triangular
        q,r = np.linalg.qr(H_inv)
        camera_rotation_est = q.T
        K_est = np.linalg.inv(r)
        np.testing.assert_array_almost_equal(K, K_est)
