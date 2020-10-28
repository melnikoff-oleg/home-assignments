#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple
from collections import defaultdict

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
    compute_reprojection_errors,
    Correspondences
)
from _corners import FrameCorners


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None,
                          max_reprojection_error: float = 1.0,
                          min_triangulation_angle_deg: float = 1.0,
                          min_depth: float = 0.1,
                          max_retiangulation_points: int = 10,
                          retriangulation_iters: int = 1000) -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    random.seed(239)

    view_mats = np.full((len(rgb_sequence),), None)
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()

    triang_params = TriangulationParameters(
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        min_depth=min_depth)

    def triangulate_and_add_points(corners1, corners2, view_mat1, view_mat2):
        points3d, ids, median_cos = triangulate_correspondences(
            build_correspondences(corners1, corners2),
            view_mat1, view_mat2,
            intrinsic_mat, triang_params)
        point_cloud_builder.add_points(ids, points3d)

    triangulate_and_add_points(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]],
                               view_mats[known_view_1[0]], view_mats[known_view_2[0]])

    def retriangulate(corner_id):
        frame_ids = []
        for i in range(len(view_mats)):
            if view_mats[i] is None:
                continue
            if corner_id in corner_storage[i].ids:
                frame_ids.append(i)
        random.shuffle(frame_ids)
        frame_ids = frame_ids[:min(len(frame_ids), max_retiangulation_points)]
        if len(frame_ids) <= 10:
            return None
        best_inliers, best_point3d = [], None

        def point2d_by_frame_id(ы):
            return corner_storage[ы].points[corner_storage[ы].ids.index(corner_id)]

        for _ in range(retriangulation_iters):
            i, j = random.choice(len(frame_ids)), random.choice(len(frame_ids))
            if i == j:
                continue
            point3d, ids, median_cos = triangulate_correspondences(
                Correspondences(
                    np.array([corner_id]),
                    np.array([point2d_by_frame_id(i)]),
                    np.array([point2d_by_frame_id(j)])),
                view_mats[i], view_mats[j], intrinsic_mat, triang_params)
            inliers = []
            for frame_id in frame_ids:
                if compute_reprojection_errors(point3d,
                                               point2d_by_frame_id([frame_id]),
                                               intrinsic_mat @ view_mats[frame_id])[0] \
                        < max_reprojection_error:
                    inliers.append(frame_id)
            if len(best_inliers) <= len(inliers):
                best_inliers, best_point3d = inliers, point3d
        return best_point3d

    time_since_last_retriangulation = defaultdict(lambda: 1e9)
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
            retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                point_cloud_builder.points[ids1],
                corner_storage[i].points[ids2],
                intrinsic_mat,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
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

            break
        if not found_new_view_mat:
            break

        # Retriangulate
        for j in range(len(point_cloud_builder.ids)):
            corner_id = point_cloud_builder.ids[j][0]
            if time_since_last_retriangulation[corner_id] < 30:
                continue
            new_pos = retriangulate(corner_id)
            if new_pos is not None:
                time_since_last_retriangulation[corner_id] = 0
                point_cloud_builder.add_points(corner_id, new_pos)

        for key in time_since_last_retriangulation.keys():
            time_since_last_retriangulation[key] += 1

    for i in list(range(0, len(view_mats))) * 2:
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]

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
