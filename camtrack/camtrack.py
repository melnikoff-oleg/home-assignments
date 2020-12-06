#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import random
import cv2
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
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
    Correspondences
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
        for j in range(i + 1, len(corner_storage)):
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
                retval, rvecs, tvecs = cv2.solvePnP(
                    points3d[inliers], points2d[inliers], intrinsic_mat,
                    rvec=rvecs, tvec=tvecs,
                    useExtrinsicGuess=True,
                    distCoeffs=None,
                    flags=cv2.SOLVEPNP_ITERATIVE)
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

        if not found_new_view_mat:
            break

    for i in range(0, len(view_mats)):
        if view_mats[i] is None:
            print('Я сдох: не все кадры обработаны :(')
            exit(1)

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
