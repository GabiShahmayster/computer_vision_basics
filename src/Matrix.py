from typing import Tuple

import numpy as np

def get_small_angles_rotation_matrix_from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    """
    This method returns a rotation matrix which corresponds to a small-angles rotation approximation
    """
    return np.eye(3) + get_skew_symmetric_matrix_representation_of_vector(vec=rotation_vector)

def get_rodrigues_rotation_matrix_from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    """
    This method returns a rotation matrix which corresponds to the rotation vector
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    rotation_mag: float = np.sqrt(rotation_vector @ rotation_vector.T)
    rotation_skew: np.ndarray = get_skew_symmetric_matrix_representation_of_vector(rotation_vector)
    return np.eye(3) + \
           np.sin(rotation_mag) / rotation_mag * rotation_skew + \
           (1-np.cos(rotation_mag)) / rotation_mag**2 * rotation_skew @ rotation_skew

def apply_small_angle_correction_to_rotation_matrix(rotation_matrix: np.ndarray, yaw_rad: float, pitch_rad: float,
                                                    roll_rad: float) -> np.ndarray:
    """
    This method applies a small-angles correction to a rotation matrix and normalizes the result
    @param yaw_rad:
    @param pitch_rad:
    @param roll_rad:
    @return:
    """
    euler_angles_correction: np.ndarray = np.array([[roll_rad],
                                                    [pitch_rad],
                                                    [yaw_rad]])
    attitude_matrix_correction: np.ndarray = get_small_angles_rotation_matrix_from_rotation_vector(euler_angles_correction)
    return normalize_rotation_matrix(rotation_matrix=attitude_matrix_correction @ rotation_matrix)

def normalize_rotation_matrix(rotation_matrix: np.ndarray, dummy: bool = True) -> np.ndarray:
    """
    get normalized rotation matrix (direction cosine matrix) to ensure orthogonality
    reference: P.Savage, Strapdown Analytics 1, (7.1.1.3-1)
    @param rotation_matrix: rotation matrix, 3X3
    @param dummy:
    @return: normalized rotation matrix, 3X3
    """
    # estimate orthogonality/normality error (7.1.1.3-1)
    Esym = 1 / 2 * (rotation_matrix @ rotation_matrix.transpose() - np.eye(3))

    # force orthogonality/normality (7.1.1.3-10)
    return (np.eye(3) - Esym) @ rotation_matrix

def get_euler_angles_YZX_order_from_rotation_matrix(rotation_matrix: np.ndarray) -> Tuple:
    R: np.ndarray = rotation_matrix # rotation matrix
    rot_y = np.arctan2(-R[2,0],R[0, 0]) # alpha = atan(-R31/R11)
    rot_z = np.arctan(R[1,0]/np.sqrt(1-R[1,0]**2)) # beta = atan(R21/sqrt(1-R21^2))
    rot_x = np.arctan2(-R[1,2],R[1, 1]) # gamma = atan(-R23/R22)
    return rot_y, rot_z, rot_x

def get_rotation_matrix_from_euler_angles_ZYX_order(heading_rad: float,
                                                    pitch_rad: float,
                                                    roll_rad: float) -> np.ndarray:
    """
    construct rotation matrix (direction cosine matrix) given 3 euler angles,
    assuming Z-Y-X order of rotation
    D.H.Titterton, Strapdown Inertial Navigation, (3.49)
    @param heading_rad: heading euler angle [rad]
    @param pitch_rad: pitch euler angle [rad]
    @param roll_rad: roll euler angle [rad]
    @return: rotation matrix, 3X3
    """

    out = np.zeros((3, 3))
    cosYaw = np.cos(heading_rad)
    cosPitch = np.cos(pitch_rad)
    cosRoll = np.cos(roll_rad)
    sinYaw = np.sin(heading_rad)
    sinPitch = np.sin(pitch_rad)
    sinRoll = np.sin(roll_rad)

    out[0][0] = cosPitch * cosYaw
    out[0][1] = -cosRoll * sinYaw + sinRoll * sinPitch * cosYaw
    out[0][2] = sinRoll * sinYaw + cosRoll * sinPitch * cosYaw

    out[1][0] = cosPitch * sinYaw
    out[1][1] = cosRoll * cosYaw + sinRoll * sinPitch * sinYaw
    out[1][2] = -sinRoll * cosYaw + cosRoll * sinPitch * sinYaw

    out[2][0] = -sinPitch
    out[2][1] = sinRoll * cosPitch
    out[2][2] = cosRoll * cosPitch
    return out

def get_vector_from_skew_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    in = [0  -z   y
          z   0  -x
          -y  x   0]
    out  = [x,y,z]
    """
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])

def get_skew_symmetric_matrix_representation_of_vector(vec: np.ndarray) -> np.ndarray:
    """
    return skew-symmetric matrix from input vector
    in  = [x,y,z]
    out = [0  -z   y
          z   0  -x
          -y  x   0]
    @param vec:
    @param dummy:
    @return: skew-symmetric matrix representation of input vector
    """
    #

    # bad input size
    assert (vec.size == 3)

    vec = vec.squeeze()
    out = np.zeros((3, 3))
    out[0][1] = -vec[2]
    out[0][2] = vec[1]

    out[1][0] = vec[2]
    out[1][2] = -vec[0]

    out[2][0] = -vec[1]
    out[2][1] = vec[0]
    return out