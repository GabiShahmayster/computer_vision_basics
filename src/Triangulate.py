from typing import Optional
import numpy as np

# def triangulate_landmark_using_bearings(translation_2_to_1: np.ndarray,
#                                         bearing_1: np.ndarray,
#                                         bearing_2_in_1: np.ndarray) -> Optional[np.ndarray]:
#     """
#     # https://apps.dtic.mil/dtic/tr/fulltext/u2/a559309.pdf
#     https://www.youtube.com/watch?v=UZlRhEUWSas&ab_channel=CyrillStachniss - geometric approach
#     inline Vector3 triangulate(const Vector3& bearing_1, const Vector3& bearing_2, const Matrix4& pose)
#     {
#         using namespace utils::rbm;
#
#         const Vector3 trans_12 = trans(inverse(pose));
#         const Vector3 bearing_2_in_1 = inverse(so3(pose)) * bearing_2;
#
#         Matrix2 A {};
#
#         A(0, 0) = bearing_1.dot(bearing_1);
#         A(1, 0) = bearing_1.dot(bearing_2_in_1);
#         A(0, 1) = -A(1, 0);
#         A(1, 1) = -bearing_2_in_1.dot(bearing_2_in_1);
#
#         const Vector2 b{bearing_1.dot(trans_12), bearing_2_in_1.dot(trans_12)};
#
#         const Vector2 lambda = A.inverse() * b;
#         const Vector3 pt_1 = lambda(0) * bearing_1;
#         const Vector3 pt_2 = lambda(1) * bearing_2_in_1 + trans_12;
#         return (pt_1 + pt_2) / 2.0;
#     @return:
#     """
#     A: np.ndarray = np.empty((2, 2))
#     A[0, 0] = bearing_1 @ bearing_1
#     A[1, 0] = bearing_1 @ bearing_2_in_1
#     A[0, 1] = -A[1, 0]
#     A[1, 1] = -bearing_2_in_1 @ bearing_2_in_1
#
#     b: np.ndarray = np.array([bearing_1 @ translation_2_to_1,
#                               bearing_2_in_1 @ translation_2_to_1])
#     try:
#         lamda: np.ndarray = np.linalg.inv(A) @ b
#         pt_1: np.ndarray = lamda[0] * bearing_1
#         pt_2: np.ndarray = lamda[1] * bearing_2_in_1 + translation_2_to_1
#         return (pt_1 + pt_2) / 2
#     except:
#         return None