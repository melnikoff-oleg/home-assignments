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
        List of corner coordinates in img1.
    """

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=pyramidal_iters,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps))
    new_poses, _, _ = cv2.calcOpticalFlowPyrLK((img0 * 255).astype(np.uint8),
                                               (img1 * 255).astype(np.uint8),
                                               poses.astype('float32').reshape((-1, 1, 2)),
                                               None,
                                               **lk_params)
    return np.empty((0, 2), dtype=np.float32) is new_poses is None else new_poses.reshape((-1, 2))


def get_eigvals_interpr2d(img, corner_size=7):
    # Returns approximation function of corners quality for 'img'.
    ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=corner_size)
    iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=corner_size)
    ix2 = ix * ix
    iy2 = iy * iy
    ixiy = ix * iy
    ix2_neighb_sum = cv2.GaussianBlur(ix2, (corner_size, corner_size), 0)
    iy2_neighb_sum = cv2.GaussianBlur(iy2, (corner_size, corner_size), 0)
    ixiy_neighb_sum = cv2.GaussianBlur(ixiy, (corner_size, corner_size), 0)
    matrices = np.dstack((ix2_neighb_sum, ixiy_neighb_sum, ixiy_neighb_sum, iy2_neighb_sum))
    matrices = matrices.reshape((matrices.shape[0], matrices.shape[1], 2, 2))
    eigvals = np.ndarray.min(np.linalg.eigvals(matrices), axis=2)
    return interp2d(np.arange(img.shape[1]),
                    np.arange(img.shape[0]),
                    eigvals)


def find_quality(eigvals_interpt2d, corner):
    return eigvals_interpt2d(corner[0], corner[1])


def detect_corners(img, mask, N=5000, block_size=7):
    # Finds corners on image 'img' with 'mask'. Returns list of corners coordinates and corners qualities.
    corners = cv2.goodFeaturesToTrack(
        img, N, 0.01, 1,
        mask=mask,
        useHarrisDetector=False,
        blockSize=block_size)
    if corners is None:
        return np.empty(shape=(0, 2), dtype=float), np.empty(shape=(0,), dtype=float)
    corners = corners[:, 0, :]
    eigvals_img = get_eigvals_interpr2d(img)
    qualities = []
    for corner in np.array(corners):
        quality = eigvals_img(corner[0], corner[1])
        qualities.append(quality)
    return corners, np.array(qualities)


def detect_new_corners_pyramidal(img, mask, pyramidal_iters=3):
    """Find corners on 'img' using 'mask' using pyramidal method.

    Returns:
         (list of all corner coordinates,
          size of each corner,
          quality of each corner)"""
    compressed_img = img.copy()
    corner_mask = mask.copy()
    k = 1
    all_corners, qualities, sizes = np.empty(shape=(0, 2), dtype=float), \
                                    np.empty(shape=(0,), dtype=float), \
                                    np.empty(shape=(0,), dtype=float)
    for _ in range(pyramidal_iters):
        corners, corners_quality = detect_corners(compressed_img, corner_mask)
        all_corners = np.concatenate((all_corners, corners * k), axis=0)
        qualities = np.concatenate((qualities, corners_quality.reshape((-1,))), axis=0)
        sizes = np.concatenate((sizes, np.repeat(3 * k, corners.shape[0])), axis=0)
        compressed_img = cv2.pyrDown(compressed_img)
        corner_mask = corner_mask[::2, ::2]
        k *= 2
    return all_corners, sizes, qualities


def get_corners_mask(corners, corner_sizes, shape):
    # Returns a mask with all corners from 'corners'. 'shape' is a shape of output mask.
    corners_mask = np.full(shape, 255).astype(np.uint8)
    for i in range(len(corners)):
        corner = corners[i].astype(np.int)
        size = corner_sizes[i]
        corners_mask = cv2.circle(corners_mask, tuple(corner), int(size), color=0, thickness=-1)
    return corners_mask


def delete_bad_corners(corners, ids, sizes, qualities, quality_threshold):
    mask = qualities > quality_threshold
    return corners[mask], ids[mask], sizes[mask], qualities[mask]


class CornersTracker:
    def __init__(self, img0, quality_threshold=200):
        self.quality_threshold = quality_threshold

        # Initialize corner properties
        self.img0 = img0
        self.corners, self.corner_sizes, self.corner_qualities = \
            detect_new_corners_pyramidal(img0, get_corners_mask(np.array([]), np.array([]), img0.shape))
        self.corners, _, self.corner_sizes, self.corner_qualities = \
            delete_bad_corners(self.corners,
                               np.arange(self.corners.shape[0]),
                               self.corner_sizes,
                               self.corner_qualities,
                               self.quality_threshold)
        self.corner_ids = np.arange(self.corners.shape[0])
        self.last_id = self.corners.shape[0] + 1

    def track_previous_corners(self, img1):
        """Returns new corners and their qualities for all
        current corners tracked on new 'img1'"""
        qualities = []
        eigvals_img1 = get_eigvals_interpr2d(img1)
        new_corners = run_optflow_pytramidal(self.img0, img1, self.corners)
        for new_corner in new_corners:
            quality = find_quality(eigvals_img1, new_corner)
            qualities.append(quality[0])
        return np.array(new_corners), np.array(qualities)

    def find_and_track_corners(self, img1):
        # Track previous corners and delete bad ones.
        prev_corners, prev_corners_quality = self.track_previous_corners(img1)
        prev_corners, self.corner_ids, self.corner_sizes, self.corner_qualities = \
            delete_bad_corners(prev_corners,
                               self.corner_ids,
                               self.corner_sizes,
                               self.corner_qualities,
                               self.quality_threshold)

        # Find new corners.
        new_corners, new_corner_sizes, new_qualities = \
            detect_new_corners_pyramidal(img1, get_corners_mask(prev_corners, self.corner_sizes, img1.shape))
        new_corners, _, new_corner_sizes, new_qualities = \
            delete_bad_corners(new_corners,
                               np.arange(new_corners.shape[0]),
                               new_corner_sizes,
                               new_qualities,
                               self.quality_threshold)

        # Concatenates old corners with new ones.
        new_ids = np.arange(self.last_id, self.last_id + new_corners.shape[0])
        self.img0 = img1
        self.corners = np.concatenate((prev_corners, new_corners), axis=0)
        self.corner_ids = np.concatenate((self.corner_ids, new_ids), axis=0)
        self.corner_sizes = np.concatenate((self.corner_sizes, new_corner_sizes), axis=0)
        self.corner_qualities = np.concatenate((self.corner_qualities, new_qualities), axis=0)
        if len(self.corner_ids) > 0:
            self.last_id = self.corner_ids[-1] + 1

    def get_frame_corners(self):
        return FrameCorners(self.corner_ids, self.corners[:], self.corner_sizes)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    tracker = CornersTracker(frame_sequence[0])
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
