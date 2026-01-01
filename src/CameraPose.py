from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation

from src.TimeUtils import Time
from src.Transformations import HomogeneousTransformation3D
import open3d as o3d

class CameraIntrinsics:
    camera_matrix: np.ndarray
    camera_distortion: Optional[np.ndarray]
    frame_width_px: int
    frame_height_px: int

    def __init__(self, camera_matrix: np.ndarray,
                 camera_distortion: Optional[np.ndarray],
                 frame_width_px: int,
                 frame_height_px: int):
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        self.frame_width_px = frame_width_px
        self.frame_height_px = frame_height_px

    @classmethod
    def from_open3d_intrinsics(cls, open3d_intrinsics: o3d.camera.PinholeCameraIntrinsic) -> 'CameraIntrinsics':
        """
        This method constructs a CameraIntrinsics object using Open3D pinhole camera intrinsics
        """
        return CameraIntrinsics(camera_matrix=open3d_intrinsics.intrinsic_matrix,
                                camera_distortion=None,
                                frame_width_px=open3d_intrinsics.width,
                                frame_height_px=open3d_intrinsics.height)

class CameraPose(HomogeneousTransformation3D):
    """
    This class specifically refers to the transformation parameters as camera pose parameters, i.e rotation and
    translation of a camera
    """

    def __copy__(self):
        return CameraPose(Rotation.from_quat(self.rotation.as_quat()),
                          self.translation.copy())

    def inverse(self) -> 'CameraPose':
        inv: HomogeneousTransformation3D = super().inverse()
        return CameraPose(rotation=inv.rotation,
                          translation=inv.translation)

class VisualOdometryData:
    """
    This object contains visual odometry data, for a specific frame
    """
    time: Time
    frame_number: int
    camera_pose: CameraPose

    def __init__(self, time: Time,
                 frame_number: int,
                 camera_pose: CameraPose):
        self.time = time
        self.frame_number = frame_number
        self.camera_pose = camera_pose

    @classmethod
    def build(cls, frame_number: int,
              time: Time,
              camera_pose: CameraPose):
        return VisualOdometryData(time,
                                  frame_number,
                                  camera_pose.__copy__())

    def __copy__(self):
        return VisualOdometryData(self.time,
                                  self.frame_number,
                                  self.camera_pose.__copy__())
