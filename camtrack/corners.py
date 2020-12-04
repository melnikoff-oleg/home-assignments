#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
from scipy.interpolate import interp2d
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def run_optflow_pytramidal(img0, img1, poses, pyramidal_iters=7, eps=1e-2, max_iter=10):
    """Finds a new coordinates of corners from 'img0' to 'img1'.

    Attributes:
        img0: previous video frame
        img1: new video frame
        poses: np.array(n, 2), list of corners coordinates in img0

    Returns:
        List of corner coordinates in img1;
        List of values (0,1): 1 iff the flow has been found.
    """

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=pyramidal_iters,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps))
    new_poses, st, _ = cv2.calcOpticalFlowPyrLK((img0 * 255).astype(np.uint8),
                                                (img1 * 255).astype(np.uint8),
                                                poses.astype('float32').reshape((-1, 1, 2)),
                                                None,
                                                **lk_params)
    if new_poses is None:
        return np.empty((0, 2), dtype=np.float32), \
               np.empty((0,), dtype=np.int)
    return new_poses.reshape((-1, 2)), st.reshape((-1,))


def detect_corners(img, mask, max_corners, min_dist, quality_level=0.003, block_size=7):
    # Finds corners on image 'img' with 'mask'. Returns list of corners coordinates.
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_dist,
        mask=mask,
        useHarrisDetector=False,
        blockSize=block_size)
    if corners is None:
        return np.empty(shape=(0, 2), dtype=float)
    return corners.reshape((-1, 2))


def detect_new_corners_pyramidal(img, prev_corners, min_dist, pyramidal_iters=3, max_corners=1000, quality_level=0.003):
    """Find corners on 'img' using pyramidal method.

    Returns:
         List of all corner coordinates;
         Size of each corner."""
    compressed_img = img.copy()
    all_corners = prev_corners.copy()
    k = 1
    new_corners, sizes = np.empty(shape=(0, 2), dtype=float), \
                         np.empty(shape=(0,), dtype=float)
    for _ in range(pyramidal_iters):
        mask = get_corners_mask(all_corners, img.shape, radius=min_dist)[::k, ::k]
        corners = detect_corners(compressed_img,
                                 mask,
                                 min_dist=min_dist,
                                 max_corners=max_corners,
                                 quality_level=quality_level)
        all_corners = np.concatenate((all_corners, corners * k), axis=0)
        new_corners = np.concatenate((new_corners, corners * k), axis=0)
        sizes = np.concatenate((sizes, np.repeat((img.shape[0] + img.shape[1]) // 500 * k, corners.shape[0])), axis=0)
        compressed_img = cv2.pyrDown(compressed_img)
        k *= 2
    return new_corners, sizes


def get_corners_mask(corners, shape, radius):
    # Returns a mask with all corners from 'corners'. 'shape' is a shape of output mask.
    corners_mask = np.full(shape, 255).astype(np.uint8)
    for i in range(len(corners)):
        corner = corners[i].astype(np.int)
        corners_mask = cv2.circle(corners_mask, tuple(corner), radius, color=0, thickness=-1)
    return corners_mask


def filter_too_close_corners(corners, st, min_dist):
    """Detects the corners which are too close to each other. Only checks corners[st == 1].
    Returns 'new_st' array such that corners[new_st == 1] is a subset of corners[st == 1]
    and the minimum distance between corners is at least 'min_dist'."""

    added_corners = np.empty((0, 2), dtype=np.float32)
    new_st = st.copy()
    for i, corner in enumerate(corners):
        if np.all(np.linalg.norm(added_corners - corner, axis=1) >= min_dist):
            added_corners = np.append(added_corners, corner.reshape((1, -1)), axis=0)
        else:
            new_st[i] = 0
    return new_st


class CornersTracker:
    def __init__(self, img0, min_dist):
        self.min_dist = min_dist

        # Initialize corner properties
        self.img0 = img0
        self.corners, self.corner_sizes = \
            detect_new_corners_pyramidal(img0, np.empty(shape=(0, 2), dtype=float), self.min_dist)
        self.corner_ids = np.arange(self.corners.shape[0])
        self.last_id = self.corners.shape[0] + 1

    def find_and_track_corners(self, img1):
        # Track previous corners and delete bad ones.
        prev_corners, st = run_optflow_pytramidal(self.img0, img1, self.corners)
        st = filter_too_close_corners(prev_corners, st, self.min_dist / 2)

        # Update bad track numbers.
        prev_corners = prev_corners[st == 1]
        prev_corner_sizes = self.corner_sizes[st == 1]
        prev_corner_ids = self.corner_ids[st == 1]

        # Find new corners.
        new_corners, new_corner_sizes = \
            detect_new_corners_pyramidal(img1, prev_corners, self.min_dist)
        new_ids = np.arange(self.last_id, self.last_id + new_corners.shape[0])

        # Concatenate old corners with new ones.
        self.img0 = img1
        self.corners = np.concatenate((prev_corners, new_corners), axis=0)
        self.corner_ids = np.concatenate((prev_corner_ids, new_ids), axis=0)
        self.corner_sizes = np.concatenate((prev_corner_sizes, new_corner_sizes), axis=0)
        if len(self.corner_ids) > 0:
            self.last_id = self.corner_ids[-1] + 1

    def get_frame_corners(self):
        return FrameCorners(self.corner_ids, self.corners, self.corner_sizes)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    tracker = CornersTracker(frame_sequence[0],
                             (frame_sequence[0].shape[0] + frame_sequence[0].shape[1]) // 300)
    builder.set_corners_at_frame(0, tracker.get_frame_corners())
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        tracker.find_and_track_corners(image_1)
        builder.set_corners_at_frame(frame, tracker.get_frame_corners())


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
