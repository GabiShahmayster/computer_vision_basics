import json
from collections import namedtuple
from pyproj import Proj, transform
import numpy as np
from typing import Type, Tuple, Optional, Union
from src.Constants import radiansToDegrees, degreesToRadians, Eccentricity, SemiMajorAxis
from src.TimeUtils import Time
from scipy.spatial.transform.rotation import Rotation

CartesianPositionParameters = namedtuple('CartesianPositionParameters', 'X Y Z')
GeographicPositionParameters = namedtuple('GeographicPosition', 'LatitudeInDegrees LongitudeInDegrees AltitudeInMeters')


class Pose:
    """
    This class contains pose information, resolved in the ECEF frame
    """

    time: Time
    translation: np.ndarray
    rotation: Rotation
    velocity: np.ndarray

    frame_number: int
    measurement_index: int

    def __init__(self, time: Time,
                 translation: np.ndarray,
                 rotation: Rotation,
                 velocity: np.ndarray = None,
                 frame_number: int = None,
                 measurement_index: int = None):
        self.time = time
        self.frame_number = frame_number
        self.measurement_index = measurement_index

        self.translation = translation
        self.rotation = rotation
        self.velocity = velocity


class HomogeneousTransformation3D:
    """
    This class represents a 3D rotation and translation (rigid motion), as a homogeneous transformation of the form:
    T(4x4) = [R(3x3) , t(3,1)]
              0(1x3) ,      1]
    and it belongs to SE(3)

    https://thydzik.com/academic/robotics-315/chap2.pdf
    """
    rotation: Rotation
    translation: np.ndarray

    def __init__(self, rotation: Rotation,
                 translation: np.ndarray):
        self.rotation = rotation
        self.translation = translation

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'HomogeneousTransformation3D':
        """
        This method returns a HomogeneousTransformation3D object, from an input homogeneous transformation 4x4 matrix
        @param matrix:
        @return:
        """
        assert (matrix.shape == (4, 4)), "HomogeneousTransformation3D can only be created from 4x4 input matrix"
        return HomogeneousTransformation3D(rotation=Rotation.from_matrix(matrix[:3, :3]),
                                           translation=matrix[:3, 3])

    def __matmul__(self, other: 'HomogeneousTransformation3D'):
        """
        define @ operator
        @param other:
        @return:
        """
        # assert(type(other) is 'HomogeneousTransformation3D')
        res: np.ndarray = self.as_se3() @ other.as_se3()
        return HomogeneousTransformation3D.from_matrix(matrix=res)

    def as_se3(self) -> np.ndarray:
        """
        This method returns the camera pose as an equivalent homogeneous transformation form
        @return:
        """
        return np.vstack((np.hstack((self.rotation.as_matrix(), self.translation.reshape((3, 1)))),
                          np.hstack((np.zeros((1, 3)), np.ones((1, 1))))))

    def get_translation_vector(self) -> np.ndarray:
        """
        This method returns the translation vector
        @return:
        """
        return self.translation.__copy__()

    def get_rotation_matrix(self) -> np.ndarray:
        """
        This method returns the rotation matrix
        @return:
        """
        return self.rotation.as_matrix()

    def __copy__(self):
        return HomogeneousTransformation3D(rotation=Rotation.from_matrix(self.rotation.as_matrix()),
                                           translation=self.translation.copy())

    def inverse(self) -> 'HomogeneousTransformation3D':
        """
        This method returns the inverse transformation of the homogeneous from
        @return:
        """
        inv_rotation: Rotation = self.rotation.inv()
        return HomogeneousTransformation3D(rotation=inv_rotation,
                                           translation=-inv_rotation.as_matrix() @ self.translation)

    def change_reference_frame(self, transformation_old_frame_to_new_frame: Type[
        'HomogeneousTransformation3D']) -> 'HomogeneousTransformation3D':
        """
        This is a similarity transformation of the original transformation M, given in
        reference frame (A), to transformation M' in reference frame (B), with T being the transformation
        from reference frame (A) to reference frame (B)
        M' = T * M * T^-1
        https://academic.csuohio.edu/richter_h/courses/mce647/mce647_2_hand.pdf

        This method changes the reference frame of an input 3D transformation and rotation
        for example, a visual-odometry based input rotation matrix of eye(3,3), which represents zero rotation w.r.t
        to the camera frame (x-vehicle right, y-vehicle down, z-vehicle forward), cannot be used to represent
        rotation w.r.t to the IMU frame (x-vehicle forward, y-vehicle right, z-vehicle down) because it would represent
        a +90[deg] heading and +90[deg] roll w.r.t to the IMU frame.
        Denoting R_cam as the rotation matrix which rotates vectors from the current camera pose to the initial camera
        frame, we have:
        x_cam0 = R_cam * x_cam
        where x_cam is a vector measured in the current camera frame and x_cam0 is the rotation of that vector to the
        initial camera frame.
        We are looking for
        x_vehicle = R_cam_to_vehicle_calc * x_cam
        where R_cam_to_vehicle is a rotation of x_cam to the vehicle frame.
        Because the rotation matrix, R_cam, represents a linear transformation of x_cam, we can consider this problem
        as a changes of basis for the transformation of x_cam, hence:
        x_vehicle = (R_cam_to_vehicle * R_cam * R_cam_to_vehicle') * x_cam
        giving us:
        R_cam_to_vehicle_calc = R_cam_to_vehicle * R_cam * R_cam_to_vehicle'
        (see https://brilliant.org/wiki/change-of-basis/)
        @param transformation:
        @param rotation_matrix_old_frame_to_new_frame:
        @return:
        """
        transformed = transformation_old_frame_to_new_frame.as_se3() \
                      @ self.as_se3() \
                      @ transformation_old_frame_to_new_frame.inverse().as_se3()
        return HomogeneousTransformation3D(rotation=Rotation.from_matrix(transformed[:3, :3]),
                                           translation=transformed[:3, 3])

    def apply_camera_motion(self, transform: Union[Type['HomogeneousTransformation3D'], np.ndarray]) -> 'HomogeneousTransformation3D':
        """
        This method rotates and then translates the camera, using homogeneous coordinates
        C_transformed = C_original * T_original_to_transformed
        @param transform:
        @return:
        """
        if type(transform) is HomogeneousTransformation3D:
            return HomogeneousTransformation3D.from_matrix(self.as_se3() @ transform.as_se3())
        elif type(transform) is np.ndarray:
            return HomogeneousTransformation3D.from_matrix(self.as_se3() @ transform)
        else:
            assert False, "wrong object type provided to apply_camera_motion"

    def get_camera_motion(self, reference_pose: Type['HomogeneousTransformation3D']) -> 'HomogeneousTransformation3D':
        """
        This method recovers the transformation from an origin camera pose to the current camera pose, recovering
        the transformation T_original_to_transformed from:
        C_transformed = C_original * T_original_to_transformed
        as T_original_to_transformed = C_original^(-1) * C_transformed
        @param camera_pose:
        @return:
        """
        T_original_to_transformed = np.linalg.inv(reference_pose.as_se3()) @ self.as_se3()
        return HomogeneousTransformation3D(rotation=Rotation.from_matrix(T_original_to_transformed[:3, :3]),
                                           translation=T_original_to_transformed[:3, 3])

    def __str__(self) -> str:
        translation: np.ndarray = self.get_translation_vector()
        rotation_y_deg, rotation_z_deg, rotation_x_deg = self.rotation.as_euler('YZX',degrees=True)
        return json.dumps({"translation x": translation[0],
                           "translation y": translation[1],
                           "translation z": translation[2],
                           "rotation y [deg]": rotation_y_deg,
                           "rotation z [deg]": rotation_z_deg,
                           "rotation x [deg]": rotation_x_deg})

def get_ecef_to_ned_rotation_matrix(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    """
    This method computes the rotation matrix from ECEF to NED for an NED reference-frame whose origin is
    located at (latitude,longitude).
    Reference: https://www.mathworks.com/help/aeroblks/directioncosinematrixeceftoned.html
    @return:
    """
    sin_mu: float = np.sin(latitude_deg * degreesToRadians)
    cos_mu: float = np.cos(latitude_deg * degreesToRadians)
    sin_l: float = np.sin(longitude_deg * degreesToRadians)
    cos_l: float = np.cos(longitude_deg * degreesToRadians)
    return np.array([[-sin_mu * cos_l, -sin_mu * sin_l, cos_mu],
                     [-sin_l, cos_l, 0],
                     [-cos_mu * cos_l, -cos_mu * sin_l, -sin_mu]], )


def get_earth_curvature_radii(lat_rad: float, alt_m: float) -> Tuple[float, float]:
    """
    Get earth horizontal radii of curvature, for current latitude and altitude
    Implemented according to "Strapdown Inertial Navigation", D.H.Titterton, paragraph 3.7.2, page 49
    @param lat_rad:
    @param alt_m:
    @return: out[0] - north-south radius of curvature, [m]
             out[1] - west-east radius of curvature, [m]
    """
    meridian_radius_of_curvature, transverse_radius_of_curvature = get_meridian_and_transverse_radii_of_curvature(
        lat_rad)
    north_south_radius: float = meridian_radius_of_curvature + alt_m
    # project transverse radius of curvature onto longitudinal plane
    west_east_radius: float = (transverse_radius_of_curvature + alt_m) * np.cos(lat_rad)
    return north_south_radius, west_east_radius


def transform_geographic_to_ecef_position(geo: GeographicPositionParameters) -> np.ndarray:
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    X, Y, Z = transform(p1=lla,
                        p2=ecef,
                        x=geo.LongitudeInDegrees,
                        y=geo.LatitudeInDegrees,
                        z=geo.AltitudeInMeters,
                        radians=False)
    return np.array([X, Y, Z])


def transform_position_ecef_to_geographic(position_ecef: np.ndarray) -> GeographicPositionParameters:
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = transform(p1=ecef,
                              p2=lla,
                              x=position_ecef[0],
                              y=position_ecef[1],
                              z=position_ecef[2])
    return GeographicPositionParameters(LatitudeInDegrees=lat,
                                        LongitudeInDegrees=lon,
                                        AltitudeInMeters=alt)


def get_meridian_and_transverse_radii_of_curvature(lat_rad: float) -> Tuple[float, float]:
    """
    Get meridian and transverse radii of curvature, at surface level, for current latitude
    Implemented according to "Strapdown Inertial Navigation", D.H.Titterton, paragraph 3.7.2, page 49
    @param lat_rad: geographic latitude
    @return: out[0] - meridian radius of curvature, [m]
             out[1] - transverse radius of curvature, [m]
    """
    meridian_radius_of_curvature: float  # radius of curvature in north-south direction
    transverse_radius_of_curvature: float  # radius of curvature in transverse plane (perpendicular to meridian @ lat)
    commonTerm: float = 1 - (Eccentricity * np.sin(lat_rad)) ** 2
    transverse_radius_of_curvature = SemiMajorAxis / np.sqrt(commonTerm)  # M
    meridian_radius_of_curvature = transverse_radius_of_curvature * (1 - Eccentricity ** 2) / commonTerm
    return meridian_radius_of_curvature, transverse_radius_of_curvature
