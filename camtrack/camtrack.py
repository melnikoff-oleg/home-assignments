#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import random
import cv2
import sortednp as snp
import scipy.optimize

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
from scipy.optimize import least_squares
import sys
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    triangulate_correspondences,
    build_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences,
    project_points,
    calc_inlier_indices,
    compute_reprojection_errors,
)
from _corners import FrameCorners

triang_params = TriangulationParameters(
    max_reprojection_error=7.5,
    min_triangulation_angle_deg=1.0,
    min_depth=0.1)


def calculate_known_views(intrinsic_mat,
                          corner_storage: CornerStorage,
                          min_correspondencies_count=100,
                          max_homography=0.7,
                          ) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    best_points_num, best_known_views = -1, ((None, None), (None, None))
    for i in range(len(corner_storage)):
        for j in range(i + 1, min(i + 40, len(corner_storage))):
            corresp = build_correspondences(corner_storage[i], corner_storage[j])
            if len(corresp[0]) < min_correspondencies_count:
                break
            E, mask = cv2.findEssentialMat(corresp.points_1, corresp.points_2,
                                           cameraMatrix=intrinsic_mat,
                                           method=cv2.RANSAC)
            mask = (mask.squeeze() == 1)
            corresp = Correspondences(corresp.ids[mask], corresp.points_1[mask], corresp.points_2[mask])
            if E is None:
                continue

            # Validate E using homography
            H, mask = cv2.findHomography(corresp.points_1,
                                         corresp.points_2,
                                         method=cv2.RANSAC)
            if np.count_nonzero(mask) / len(corresp.ids) > max_homography:
                continue

            R1, R2, T = cv2.decomposeEssentialMat(E)
            for view_mat2 in [np.hstack((R, t)) for R in [R1, R2] for t in [T, -T]]:
                view_mat1 = np.eye(3, 4)
                points, _, _ = triangulate_correspondences(corresp, view_mat1, view_mat2,
                                                           intrinsic_mat, triang_params)
                print('Try frames {}, {}: {} correspondent points'.format(i, j, len(points)))
                if len(points) > best_points_num:
                    best_known_views = ((i, view_mat3x4_to_pose(view_mat1)), (j, view_mat3x4_to_pose(view_mat2)))
                    best_points_num = len(points)
                    if best_points_num > 1500:
                        return best_known_views
    return best_known_views


def bundle_adjustment(view_mats, point_cloud_builder, intrinsic_mat, corner_storage):
    ids2d, ids3d, frameids = [], [], []
    for frame in range(len(view_mats)):
        _, (id1, id2) = snp.intersect(point_cloud_builder.ids.flatten(),
                                      corner_storage[frame].ids.flatten(), indices=True)
        inliers = calc_inlier_indices(point_cloud_builder.points[id1],
                                      corner_storage[frame].points[id2],
                                      intrinsic_mat @ view_mats[frame], 1.0)
        for i in inliers:
            ids2d.append(id2[i])
            ids3d.append(id1[i])
            frameids.append(frame)

    view_mats_vec = []
    for view_mat in view_mats:
        tvec = view_mat[:, 3]
        rvec, _ = cv2.Rodrigues(view_mat[:, :3])
        view_mats_vec.append(np.concatenate([rvec.squeeze(), tvec]))
    view_mats_vec = np.array(view_mats_vec)

    used_3ds = list(set(ids3d))
    points3d_vec = np.array(point_cloud_builder.points[used_3ds])
    for i in range(len(ids3d)):
        ids3d[i] = used_3ds.index(ids3d[i])

    n, m = len(view_mats_vec.reshape(-1)), len(points3d_vec.reshape(-1))

    def calc_vec_errors(vec, point2d):
        r, t = vec[0:3], vec[3:6]
        view_mat, point3d = rodrigues_and_translation_to_view_mat3x4(r.reshape(3, 1),
                                                                     t.reshape(3, 1)), vec[6:9]
        return compute_reprojection_errors(point3d.reshape(1, -1),
                                           point2d.reshape(1, -1),
                                           intrinsic_mat @ view_mat)[0]

    def calc_jacobian():
        J = np.zeros((len(frameids), n + m))
        for i in range(len(frameids)):
            frameid, id3d, id2d = frameids[i], ids3d[i], ids2d[i]
            point3d = points3d_vec[id3d]
            point2d = corner_storage[frameid].points[id2d]
            vec = np.concatenate([view_mats_vec[frameid], point3d])
            der = scipy.optimize.approx_fprime(vec, lambda vec: calc_vec_errors(vec, point2d),
                                               np.full(vec.shape, 1e-7))
            J[i, n + 3 * id3d: n + 3 * id3d + 3] = der[6:]
            J[i, 6 * frameid: 6 * frameid + 6] = der[:6]
        return J

    def calc_errors():
        errors = []
        for i in range(len(frameids)):
            frameid, id3d, id2d = frameids[i], ids3d[i], ids2d[i]
            errors.append(calc_vec_errors(np.concatenate([view_mats_vec[frameid],
                                                          points3d_vec[id3d]]),
                                          corner_storage[frameid].points[id2d]))
        return errors

    error_before_bundle_adjustment = np.mean(calc_errors())

    for iter in range(5):
        J = calc_jacobian()
        u = np.array(calc_errors())
        g = J.T @ u
        gx, gc = g[n:], g[:n]
        Q = J.T @ J + 10 * np.diag(np.diag(J.T @ J))
        U, W, V = Q[:n, :n], Q[:n, n:], Q[n:, n:]
        try:
            V_inv = np.linalg.inv(V)
            deltac = np.linalg.solve(U - W @ V_inv @ W.T, W @ V_inv @ gx - gc)
            deltax = -V_inv @ (gx + W.T @ deltac)
            view_mats_vec = (view_mats_vec.reshape(-1) + deltac).reshape((-1, 6))
            points3d_vec = (points3d_vec.reshape(-1) + deltax).reshape((-1, 3))
        except:
            pass
        print('Bundle-adjustment: iteration {}, error {}'.format(iter, np.mean(calc_errors())))
        sys.stdout.flush()

    error_after_bundle_adjustment = np.mean(calc_errors())

    if error_after_bundle_adjustment > error_before_bundle_adjustment:
        print('Better without bundle adjustment')
        return view_mats

    point_cloud_builder.update_points(point_cloud_builder.ids[used_3ds], points3d_vec)
    new_view_mats = []
    for vec in view_mats_vec:
        rvec, tvec = vec[0:3], vec[3:6]
        view_mat = rodrigues_and_translation_to_view_mat3x4(rvec.reshape(3, 1), tvec.reshape(3, 1))
        new_view_mats.append(view_mat)
    return np.array(new_view_mats)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = calculate_known_views(intrinsic_mat, corner_storage)
        print('Calculated known views: {} and {}'.format(known_view_1[0], known_view_2[0]))

    random.seed(239)

    view_mats = np.full((len(rgb_sequence),), None)
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()

    def triangulate_and_add_points(corners1, corners2, view_mat1, view_mat2):
        points3d, ids, median_cos = triangulate_correspondences(
            build_correspondences(corners1, corners2),
            view_mat1, view_mat2,
            intrinsic_mat, triang_params)
        point_cloud_builder.add_points(ids, points3d)

    triangulate_and_add_points(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]],
                               view_mats[known_view_1[0]], view_mats[known_view_2[0]])

    inliers_points3d, inliers_points2d = None, None

    def find_errs(rtvecs):
        rvecs, tvecs = rtvecs[:3], rtvecs[3:]
        rmat, _ = cv2.Rodrigues(np.expand_dims(rvecs, axis=1))
        rmat = np.concatenate((rmat, np.expand_dims(tvecs, axis=1)), axis=1)
        return (project_points(inliers_points3d, intrinsic_mat @ rmat) - inliers_points2d).flatten()

    while True:
        random_ids = list(range(len(view_mats)))
        random.shuffle(random_ids)
        found_new_view_mat = False
        for i in random_ids:
            if view_mats[i] is not None:
                continue
            common, (ids1, ids2) = snp.intersect(point_cloud_builder.ids.flatten(),
                                                 corner_storage[i].ids.flatten(),
                                                 indices=True)
            if len(common) <= 10:
                continue

            points3d = point_cloud_builder.points[ids1]
            points2d = corner_storage[i].points[ids2]
            retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                points3d, points2d, intrinsic_mat,
                iterationsCount=108,
                reprojectionError=triang_params.max_reprojection_error,
                distCoeffs=None,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP)
            if retval:
                # M-оценки
                inliers_points3d = points3d[inliers.flatten()]
                inliers_points2d = points2d[inliers.flatten()]
                rtvecs = least_squares(find_errs,
                                       x0=np.concatenate((rvecs, tvecs)).flatten(),
                                       loss='huber', method='trf').x
                rvecs = rtvecs[:3].reshape((-1, 1))
                tvecs = rtvecs[3:].reshape((-1, 1))
            if not retval:
                continue

            print('Iteration {}/{}, processing {}th frame: {} inliers, {} points in point cloud'
                  .format(len([v for v in view_mats if v is not None]) - 1,
                          len(rgb_sequence) - 2, i, len(inliers),
                          len(point_cloud_builder.points)))

            view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvecs, tvecs)
            found_new_view_mat = True

            # Add new points
            inliers = np.array(inliers).astype(int).flatten()
            inlier_corners = FrameCorners(*[c[inliers] for c in corner_storage[i]])
            for j in range(len(view_mats)):
                if view_mats[j] is not None:
                    triangulate_and_add_points(corner_storage[j], inlier_corners,
                                               view_mats[j], view_mats[i])

            sys.stdout.flush()

        if not found_new_view_mat:
            break

    for i in range(0, len(view_mats)):
        if view_mats[i] is None:
            print('Я сдох: не все кадры обработаны :(')
            exit(1)

    if len(view_mats) < 100:  # Иначе долго работает
        view_mats = bundle_adjustment(view_mats, point_cloud_builder, intrinsic_mat, corner_storage)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
