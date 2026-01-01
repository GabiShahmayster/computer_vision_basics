from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from src.Constants import radiansToDegrees
from src.Matrix import get_skew_symmetric_matrix_representation_of_vector


def get_JPL_quaterion_norm(q: np.ndarray) -> float:
    return np.sqrt(q[0]**2 + q[1]**2 + q[2]**2+q[3]**2)

def get_JPL_quaterion_from_euler_angles_ZYX_order(heading_rad: float,
                                                  pitch_rad: float,
                                                  roll_rad: float) -> np.ndarray:
    """
    return JPL convention quaternion
    q = [q1 q2 q3 q4] = q4+i*q1+j*q2+k*q3
    according to "Indirect Kalman Filter for attitude estimation", Stergios I. Roumeliotis
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    @param rotation_matrix:
    @return:
    """
    cy: float = np.cos(heading_rad * 0.5)
    sy: float = np.sin(heading_rad * 0.5)
    cp: float = np.cos(pitch_rad * 0.5)
    sp: float = np.sin(pitch_rad * 0.5)
    cr: float = np.cos(roll_rad * 0.5)
    sr: float = np.sin(roll_rad * 0.5)
    # Quaternion q;
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def get_JPL_quaternion_equivalent_rotation_angle_in_degrees(q: np.ndarray) -> float:
    q_vec = get_JPL_quaternion_vector_part(q)
    q_real = get_JPL_quaternion_scalar_part(q)
    return 2 * np.arctan2(np.sqrt(q_vec @ q_vec.T), q_real) * radiansToDegrees

def get_JPL_quaternion_vector_part(q: np.ndarray) -> np.ndarray:
    """
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 3
    @param q:
    @return:
    """
    return q[:3]

def get_JPL_quaternion_scalar_part(q: np.ndarray) -> float:
    """
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 3
    @param q:
    @return:
    """
    return q[3]

def get_JPL_inverse_quaternion(q: np.ndarray) -> np.ndarray:
    """
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 21
    @param q:
    @return:
    """
    return np.array([-q[0],
                     -q[1],
                     -q[2],
                     q[3]])

def get_JPL_SIGMA_matrix(p: np.ndarray) -> np.ndarray:
    """
    This method returns an auxilliary SIGMA matrix, for the quaternion q
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 17
    @param q:
    @return: 4x3 SIGMA matrix
    """
    p_vec = get_JPL_quaternion_vector_part(p)
    return np.vstack((p[3] * np.eye(3) + get_skew_symmetric_matrix_representation_of_vector(p_vec),
                      -p_vec.T))


def get_JPL_rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 62
    @param q:
    @return:
    """
    q_vec: np.ndarray = get_JPL_quaternion_vector_part(q)
    q4: float = q[3]
    return (2*q4**2-1)*np.eye(3)-\
           2*q4*get_skew_symmetric_matrix_representation_of_vector(q_vec)+\
           2*q_vec.reshape((3,1))*q_vec.reshape((1,3))

def get_JPL_euler_angles_from_quaternion(q: np.ndarray) -> Tuple:
    """
    @param q:
    @return:
    """
    return Rotation.from_matrix(get_JPL_rotation_matrix_from_quaternion(q=q)).as_euler('ZYX')


def get_JPL_relative_quaternion_between_rotation_matrices(rot_1: np.ndarray,
                                                          rot_2: np.ndarray) -> np.ndarray:
    """
    This method returns the relative orientation, between two frames represented by rotation matrices,
    in quaternion form
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 18
    @param rot_1:
    @param rot_2:
    @return: q1(C1) * q2(C2)^(-1)
    """
    return multiply_JPL_quaternions(get_JPL_quaternion_from_rotation_matrix(rotation_matrix=rot_1),
                                    get_JPL_inverse_quaternion(
                                        get_JPL_quaternion_from_rotation_matrix(rotation_matrix=rot_2)))


def multiply_JPL_quaternions(q: np.ndarray,
                             p: np.ndarray) -> np.ndarray:
    """
    "Indirect Kalman Filter for 3D Attitude Estimation", eq. 7
    q x p = [qX] * p
    @param q_1:
    @param q_2:
    @return:
    """
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    return np.array([q4 * p1 + q3 * p2 - q2 * p3 + q1 * p4,
                     -q3 * p1 + q4 * p2 + q1 * p3 + q2 * p4,
                     q2 * p1 - q1 * p2 + q4 * p3 + q3 * p4,
                     -q1 * p1 - q2 * p2 - q3 * p3 + q4 * p4])


def get_JPL_quaternion_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    heading_rad, pitch_rad, roll_rad = Rotation.from_matrix(rotation_matrix).as_euler('ZYX',degrees=False)
    return get_JPL_quaterion_from_euler_angles_ZYX_order(heading_rad=heading_rad,
                                                         pitch_rad=pitch_rad,
                                                         roll_rad=roll_rad)


