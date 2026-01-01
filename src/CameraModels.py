from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from cv2 import Rodrigues, projectPoints, line, calibrationMatrixValues, ellipse, LINE_AA, putText, FONT_ITALIC
from typing import List, Tuple, Union, Type, Optional

from scipy.spatial.transform import Rotation

from src.CameraPose import CameraIntrinsics, CameraPose
from src.Constants import degreesToRadians, radiansToDegrees
from src.Matrix import get_rotation_matrix_from_euler_angles_ZYX_order, \
    get_skew_symmetric_matrix_representation_of_vector
from src.Transformations import CartesianPositionParameters
from src.Utilities import check_subclass

copy_namedtuple = lambda tuple: tuple._replace()

PinholeCameraParameters = namedtuple('PinholeCameraParameters', ['frame_width_px',
                                                                 'frame_height_px',
                                                                 'focal_length_mm',
                                                                 'sensor_width_mm',
                                                                 'sensor_height_mm'])

ExtrinsicParametersUncertainty = namedtuple('ExtrinsicParametersUncertainty', ['rotation_error_covariance_matrix',
                                                                               'translation_error_covariance_matrix'])
CalibrationMatrixValuesResult = namedtuple('CalibrationMatrixValuesResult', ['hFOV_deg',
                                                                             'vFOV_deg',
                                                                             'focal_length_mm',
                                                                             'principal_point_mm',
                                                                             'aspect_ratio'])


class PinholeCamera:
    _parameters: PinholeCameraParameters

    focal_length_x_px: int
    focal_length_y_px: int
    aspect_ratio: float

    _intrinsic_parameters: CameraIntrinsics
    camera_pose: CameraPose
    _extrinsic_parameters_uncertainty: ExtrinsicParametersUncertainty

    def __init__(self,
                 parameters: Union[PinholeCameraParameters, CameraIntrinsics],
                 camera_pose: CameraPose):
        self.update_intrinsic(parameters=parameters)
        self._update_extrinstic(camera_pose=camera_pose)

    def update_extrinsic_uncertainty(self, extrinsic_parameters_uncertainty: ExtrinsicParametersUncertainty):
        self._extrinsic_parameters_uncertainty = extrinsic_parameters_uncertainty

    def get_intrinsic(self) -> CameraIntrinsics:
        return self._intrinsic_parameters

    def _update_intrinsic(self, parameters: PinholeCameraParameters):
        self.focal_length_x_px = np.floor(
            parameters.focal_length_mm / parameters.sensor_width_mm * parameters.frame_width_px)

        self.aspect_ratio = parameters.frame_width_px / parameters.frame_height_px
        sensor_height_mm = parameters.sensor_height_mm
        if sensor_height_mm is None:
            sensor_height_mm = np.floor(parameters.sensor_width_mm / self.aspect_ratio)

        self._parameters = parameters._replace(sensor_height_mm=sensor_height_mm)

        self.focal_length_y_px = np.floor(
            parameters.focal_length_mm / self._parameters.sensor_height_mm * parameters.frame_height_px)
        self._intrinsic_parameters = CameraIntrinsics(camera_matrix=self.get_camera_matrix(),
                                                         camera_distortion=None,
                                                         frame_width_px=parameters.frame_width_px,
                                                         frame_height_px=parameters.frame_height_px)

    def get_calibration_matrix_values(self) -> CalibrationMatrixValuesResult:
        """
        This method calls OpenCV's calibrationMatrixValues method
        @return:
        """
        a = calibrationMatrixValues(self.get_camera_matrix(),
                                    imageSize=(
                                        self._intrinsic_parameters.frame_width_px,
                                        self._intrinsic_parameters.frame_height_px),
                                    apertureWidth=self._parameters.sensor_width_mm,
                                    apertureHeight=self._parameters.sensor_height_mm)
        return CalibrationMatrixValuesResult(hFOV_deg=a[0],
                                             vFOV_deg=a[1],
                                             focal_length_mm=a[2],
                                             principal_point_mm=a[3],
                                             aspect_ratio=a[4])

    def update_intrinsic(self, parameters: Union[PinholeCameraParameters, CameraIntrinsics]):
        if check_subclass(parameters, PinholeCameraParameters):
            self._update_intrinsic(parameters=parameters)
        elif check_subclass(parameters, CameraIntrinsics):
            self._intrinsic_parameters = parameters
            self.aspect_ratio = parameters.frame_width_px / parameters.frame_height_px
            SENSOR_WIDTH_MM: int = 6
            self.focal_length_x_px = parameters.camera_matrix[0][0]
            self.focal_length_y_px = parameters.camera_matrix[1][1]
            self._parameters = PinholeCameraParameters(frame_width_px=parameters.frame_width_px,
                                                       frame_height_px=parameters.frame_height_px,
                                                       focal_length_mm=parameters.camera_matrix[0][0],
                                                       sensor_width_mm=SENSOR_WIDTH_MM, # TODO - hard-coded to keep get_calibration_matrix_values() from braking
                                                       sensor_height_mm=np.floor(SENSOR_WIDTH_MM / self.aspect_ratio))

    def _update_extrinstic(self, camera_pose: CameraPose):
        self.camera_pose = camera_pose.__copy__()

    def get_camera_rotation_matrix(self) -> np.ndarray:
        return self.camera_pose.rotation.as_matrix()

    def get_camera_translation_vector(self) -> np.ndarray:
        return self.camera_pose.get_translation_vector().reshape((3, 1))

    def get_camera_matrix(self) -> np.ndarray:
        # https://support.geocue.com/converting-focal-length-from-pixels-to-millimeters-to-use-in-bentley-context-capture/
        return np.array([[self.focal_length_x_px, 0, self._parameters.frame_width_px / 2],
                         [0, self.focal_length_y_px, self._parameters.frame_height_px / 2],
                         [0, 0, 1]])

    def project_points(self,
                       points: np.ndarray,
                       cam_rotation_matrix: np.ndarray,
                       cam_translation_vector: np.ndarray,
                       cam_intrinsic: np.ndarray,
                       cam_distortion_coeff: np.ndarray) -> np.ndarray:
        image_points, jacobian = projectPoints(points,
                                               np.zeros((3, 1)).astype(np.float64),
                                               np.zeros((3, 1)).astype(np.float64),
                                               cam_intrinsic,
                                               cam_distortion_coeff)
        return image_points[:, 0, :]

    # def get_projected_points_analytical(self, list_of_objects: List[ProjectableObject]) -> np.ndarray:
    #     return self.project_points(self.get_points_to_project(list_of_objects))

    def get_projection_matrix(self) -> np.ndarray:
        # extrinsic paramaters
        R = self.camera_pose.rotation.as_matrix().T
        T = (-1) * self.camera_pose.rotation.as_matrix().T @ self.camera_pose.translation

        # build projection matrix
        return self._intrinsic_parameters.camera_matrix @ \
               np.hstack((R, T.reshape((3, 1))))

    def return_valid_image_points(self, points) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method return points which are within the FOV of the camera
        @param points:
        @return:
        """
        valid_points: np.ndarray = np.ones(points.shape[1], dtype=bool)

        # separate points before/behind camera
        world_to_camera_transformation: np.ndarray = np.linalg.inv(self.camera_pose.as_se3())
        points_camera_frame: np.ndarray = world_to_camera_transformation[:3, :3] @ points \
                                          + world_to_camera_transformation[:3, 3].reshape(3,1)

        # discard objects behind the camera (which have negative Z coordinates)
        points_before_camera: np.ndarray = valid_points & (points_camera_frame[2, :] > 0)

        if len(np.where(points_before_camera)[0]) == 0:
            return np.array([]), np.array([])

        image_points = self.project_points(points=points_camera_frame[:, valid_points].T,
                                           cam_rotation_matrix=np.eye(3),
                                           cam_translation_vector=np.zeros(3),
                                           cam_intrinsic=self._intrinsic_parameters.camera_matrix,
                                           cam_distortion_coeff=self._intrinsic_parameters.camera_distortion)

        # keep only points within frame
        points_in_fov = valid_points[points_before_camera] \
                        & (image_points[:, 0] < self._parameters.frame_width_px) \
                        & (image_points[:, 0] >= 0) \
                        & (image_points[:, 1] < self._parameters.frame_height_px) \
                        & (image_points[:, 1] >= 0)
        valid_points_indices = np.where(points_in_fov)[0]
        if len(valid_points_indices) == 0:
            return np.array([]), np.array([])
        else:
            return image_points[valid_points_indices, :], valid_points_indices

    def draw_projected_point_uncertainty_ellipse(self, frame: np.ndarray, point) -> np.ndarray:
        """
        This method overlays the input frame, with a 95% uncertainty ellipse
        @param frame:
        @param point:
        @return:
        """

        covariance_analytical: np.ndarray = self.get_projected_point_covariance(point=point)

        vals, vecs = np.linalg.eigh(5.9915 * covariance_analytical)
        indices = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

        point_projection = self.get_projected_points([point])
        if len(point_projection[1]) > 0:
            center = (int(point_projection[0].squeeze()[0]),
                      int(point_projection[0].squeeze()[1]))
            axes = int(vals[0] + .5), int(vals[1] + .5)
            angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
            ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), 2)
        return frame

    def get_projected_point_covariance(self, point) -> np.ndarray:
        """
        This method calculates the error covariance of the projection of a point, due to camera pose errors
        @param point:
        @return:
        """
        point_covariance: np.ndarray = np.zeros((3, 3))
        # assuming that camera position & attitude errors are independent

        point_covariance += self._get_projected_point_covariance_due_to_camera_position_noise(point=point)
        point_covariance += self._get_projected_point_covariance_due_to_camera_attitude_noise(point=point)
        return point_covariance

    def _get_projected_point_covariance_due_to_camera_position_noise(self, point) -> np.ndarray:
        """
        This method provides an analytical approximation to the covariance of the projection error, due to
        camera position uncertainty, modeled as a Gaussian error
        see analytical derivation in Confluence page:
        https://confluence.harman.com/confluence/display/INTINRD/Pose+Accuracy+Effect+on+AR
        @param point:
        @return:
        """
        Sigma_translation: np.ndarray = self._extrinsic_parameters_uncertainty.translation_error_covariance_matrix
        if Sigma_translation is None:
            return np.zeros((3, 3))

        M = np.array([point.center.X,
                      point.center.Y,
                      point.center.Z]).reshape((3, 1))
        Z = M[2]

        var_Z = Sigma_translation[2, 2]
        K = self.get_intrinsic().camera_matrix
        A = np.array([0, 0, 1]).reshape((1, 3))
        return 1 / Z ** 2 * K @ (Sigma_translation \
                                 + 1 / Z ** 2 * var_Z * M @ M.transpose()
                                 - 1 / Z * (
                                         Sigma_translation @ A.transpose() @ M.transpose() + M @ A @ Sigma_translation)) @ K.transpose()

    def _get_projected_point_covariance_due_to_camera_attitude_noise(self, point) -> Optional[np.ndarray]:
        """
        This method provides an analytical approximation to the covariance of the projection error, due to
        camera attitude uncertainty, modeled as a Gaussian error
        see analytical derivation in Confluence page:
        https://confluence.harman.com/confluence/display/INTINRD/Pose+Accuracy+Effect+on+AR
        @param point:
        @return:
        """
        Sigma_rotation: np.ndarray = self._extrinsic_parameters_uncertainty.rotation_error_covariance_matrix
        if Sigma_rotation is None:
            return np.zeros((3, 3))
        R: np.ndarray = self.get_camera_rotation_matrix()
        M: np.ndarray = np.array([point.center.X,
                                  point.center.Y,
                                  point.center.Z]).reshape((3, 1))
        M_skew: np.ndarray = get_skew_symmetric_matrix_representation_of_vector(M)
        K: np.ndarray = self.get_camera_matrix()
        A: np.ndarray = np.array([0, 0, 1]).reshape((1, 3))
        scale: np.ndarray = A @ K @ R @ M
        return 1 / scale ** 2 * K @ R @ M_skew @ Sigma_rotation @ M_skew.transpose() @ R.transpose() @ K.transpose()

class AnalyticalPinholeCamera(PinholeCamera):
    """
    This class overrides OpenCV's computations with analytical calculations of a pinhole camera model
    """

    def project_points_homogeneous(self, object_points: np.ndarray) -> np.ndarray:
        """
        perspective projection, s*m=K[R|T]*M
        based on https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        :param object_points:
        :return:
        """

        # rotate points to camera-frame
        N_points: int = object_points.shape[1]
        object_points_homogenous: np.ndarray = np.vstack((object_points, np.ones((1, N_points))))

        # project
        return self.get_projection_matrix() @ object_points_homogenous

    def project_points(self,
                       points: np.ndarray,
                       cam_rotation_matrix: np.ndarray,
                       cam_translation_vector: np.ndarray,
                       cam_intrinsic: np.ndarray,
                       cam_distortion_coeff: np.ndarray) -> np.ndarray:
        points = points.transpose()
        N_points: int = points.shape[0]
        projected_points: np.ndarray = np.ndarray((2, N_points))
        project_points_homogeneous: np.ndarray = self.project_points_homogeneous(object_points=points.transpose())

        # normalize
        for point_idx in range(N_points):
            projected_points[:, point_idx] = project_points_homogeneous[:2, point_idx] / project_points_homogeneous[
                2, point_idx]

        # return in the shape as OpenCV
        return projected_points.transpose().reshape((N_points, 2))

class TransformedPinholeCameraBase(ABC, PinholeCamera):
    """
    This class abstracts a pinhole camera translated and rotated w.r.t to a standard pinhole camera
    """
    rotation_matrix_to_standard_camera: np.ndarray
    translation_vector_to_standard_cam_in_cam_frame: np.ndarray

    def __init__(self, parameters: Union[PinholeCameraParameters, CameraIntrinsics],
                 camera_pose: CameraPose,
                 rotation_matrix_to_standard_camera: np.ndarray,
                 translation_vector_to_standard_cam_in_cam_frame: np.ndarray):
        """

        @param parameters:
        @param camera_pose:
        @param attitude:
        @param rotation_matrix_to_standard_camera:
        @param translation_vector_to_standard_cam_in_cam_frame:
        """

        self.rotation_matrix_to_standard_camera = rotation_matrix_to_standard_camera
        self.translation_vector_to_standard_cam_in_cam_frame = translation_vector_to_standard_cam_in_cam_frame

        PinholeCamera.__init__(self, parameters=parameters,
                               camera_pose=camera_pose)

    def _update_extrinstic(self, camera_pose: CameraPose):
        """
        This method rotates the provided extrinsic attitude parameters to the standard camera frame
        @param position:
        @param attitude:
        @return:
        """
        # update rotation matrix
        rotation_matrix_world_to_camera: np.ndarray = camera_pose.rotation.as_matrix().transpose()
        rotation_matrix_world_to_standard_camera: np.ndarray = self.rotation_matrix_to_standard_camera @ \
                                                               rotation_matrix_world_to_camera
        translation_vector_in_world_frame = rotation_matrix_world_to_standard_camera.transpose() @ (
            -self.translation_vector_to_standard_cam_in_cam_frame)
        # update camera position
        translation: np.ndarray = camera_pose.translation
        new_pose: CameraPose = CameraPose(rotation=Rotation.from_matrix(rotation_matrix_world_to_standard_camera),
                                          translation=translation+translation_vector_in_world_frame)
        super()._update_extrinstic(camera_pose=new_pose)


class RotatedPinholeCameraBase(TransformedPinholeCameraBase):
    """
    This class abstracts a pinhole camera rotated w.r.t to a standard pinhole camera, with zero translation
    """
    translation_vector_to_standard_cam_in_cam_frame = np.zeros(3)

    def __init__(self, parameters: Union[PinholeCameraParameters, CameraIntrinsics],
                 camera_pose: CameraPose):
        PinholeCamera.__init__(self, parameters=parameters,
                               camera_pose=camera_pose)


class BlenderPinholeCamera(RotatedPinholeCameraBase):
    """
    This class rotates Blender's reference frame to standard pinhole camera reference frame
    standard frame: X - camera frame right, Y - camera frame down, Z - camera frame forward
    blender frame: X - camera frame right, Y - camera frame up, Z - camera frame backward
    """

    rotation_matrix_to_standard_camera: np.ndarray = np.array([[1.0, 0, 0],
                                                               [.0, -1.0, 0],
                                                               [.0, 0, -1.0]])


class FrontFacingVehiclePinholeCamera(TransformedPinholeCameraBase):
    """
    This class implements a standard pinhole camera model, with the camera assumed to be mounted on a vehicle,
    in a front-facing orientation, so that:
    camera X-axis (horizontal direction of image) is aligned with the vehicle right Y-axis
    camera Y-axis (vertical direction of image) is aligned with the vehicle down Z-axis
    camera Z-axis (viewing direction) is aligned with the vehicle forward X-axis
    """
    rotation_matrix_to_standard_camera = np.array([[.0, 1.0, .0],
                                                   [.0, .0, 1.0],
                                                   [1.0, .0, .0]])

    def __init__(self, parameters: Union[PinholeCameraParameters, CameraIntrinsics],
                 camera_pose: CameraPose,
                 translation_vector_to_standard_cam_in_cam_frame: np.ndarray = None):
        """

        @param parameters:
        @param position:
        @param attitude:
        @param translation_vector_to_standard_camera, in vehicle-frame forward-right-down coordinates:
        """
        if translation_vector_to_standard_cam_in_cam_frame is None:
            translation_vector_to_standard_cam_in_cam_frame = np.zeros(3)
        super().__init__(parameters=parameters,
                         camera_pose=camera_pose,
                         rotation_matrix_to_standard_camera=self.rotation_matrix_to_standard_camera,
                         translation_vector_to_standard_cam_in_cam_frame=translation_vector_to_standard_cam_in_cam_frame)
