from typing import Iterable, Union

import os
from pathlib import Path

import cv2
import numpy as np

import torch


def read_image(image_path, normalize: bool = True, as_tensor: bool = False):
    """ Read an Image file and preprocess for general purpose analysis

    Note: If image file can't be found, returns None.

    :param image_path: Union[str, Path]: Path to image file
    :param normalize: bool: Normalize frame from [0, 255] to [0, 1]
    :param as_tensor: bool: Convert np.array to Torch.Tensor

    :return: Union[None, np.array, torch.Tensor]
    """

    if os.path.exists(image_path):
        image = cv2.imread(f'{image_path}', flags=cv2.IMREAD_UNCHANGED)

        if len(image.shape) > 2:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if normalize:
            image = image.astype(np.float64) / 255.

        if as_tensor:
            return torch.from_numpy(image)

        return image

    else:
        raise FileNotFoundError(f"The file '{image_path}' cannot be found")


def save_atlas(image: np.array, save_path: Union[str, Path]):
    """ Save an Output Atlas to Disk. Can be Static or Animated.

    :param image: np.array: shape (width, height, 3) or (width, height, 3, num_frames)
    :param save_path: str: Path to save the Atlas files
    """

    if len(image.shape) == 4:
        os.makedirs(save_path, exist_ok=True)

        for idx, atlas in enumerate(image):
            atlas = cv2.cvtColor(atlas, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(Path(save_path) / f"{idx:05}.png"), atlas)

    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(save_path), image)


def read_atlas(path: Path):
    """ Read Atlas from Disk. Can be Static or Animated.

    :param path: Union[str, Path]: Path to Atlas files
    :return: np.array: shape (width, height, 3) or (width, height, 3, num_frames)
    """

    if path is not None:
        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            atlas_files = sorted(path.glob("*"))

            if not atlas_files:
                raise FileNotFoundError(f"Could not find any files inside the atlas folder '{path}'")

            atlases = []

            for file in atlas_files:
                atlas = read_image(file)
                atlases.append(atlas)

            atlases = np.array(atlases)
            atlases = torch.from_numpy(atlases)

            return atlases

        else:
            return read_image(str(path), as_tensor=True)

    else:
        return None


def save_video(frames: np.array, save_path: Union[str, Path], fps: int = 10):
    """ Save a Frame Array as a MP4 Video File to Disk.

    :param frames: np.array: shape (width, height, 3, num_frames)
    :param save_path: str: Path to save the output video file
    :param fps: FPS number to render the video
    """

    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, frames.shape[:2][::-1])

    for frame_idx in range(frames.shape[3]):
        frame = cv2.cvtColor(frames[:, :, :, frame_idx], cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()


def resize_flow(flow: np.array, new_h: int, new_w: int):
    """ Resize an Optical Flow Array to a Desired Size

    :param flow: np.array: Optical Flow Array
    :param new_h: int: Desired height size
    :param new_w: int: Desired width size

    :return: flow: np.array: Resized Optical Flow Array
    """

    old_h, old_w = flow.shape[0:2]

    flow = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    flow[:, :, 0] *= new_h / old_h
    flow[:, :, 1] *= new_w / old_w

    return flow


def compute_consistency(forward_flow: np.array, backward_flow: np.array):
    """ Compute an Optical Flow Forward/Backward Consistency

    :param forward_flow: np.array: Forward Optical Flow Array
    :param backward_flow: np.array: Backward Optical Flow Array
    :return: diff: np.array: Array with consistency values for each vector
    """

    warped_backward_flow = warp_flow(backward_flow, forward_flow)
    diff = forward_flow + warped_backward_flow
    diff = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) ** .5
    return diff


def warp_flow(image: np.array, flow: np.array):
    """ Warp an Optical Flow Array given an Image

    :param image: np.array: Input Image
    :param flow: np.array: Input Optical Flow Array
    :return: res: np.array: Output Warped Optical Flow Array
    """

    h, w = flow.shape[:2]

    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(image, flow, None, cv2.INTER_LINEAR)

    return res


def get_pixel_coordinates(video_shape: torch.Size):
    """ Index Pixel Coordinates given a Video Shape

    :param video_shape: torch.Size (size_y, size_x, 3, num_frames)
    :return: pixel_coords: torch.Tensor (num_coords, 3)
    """

    coordinates = []

    for frame_idx in range(video_shape[3]):
        mask = torch.ones(video_shape[:2], dtype=torch.bool)
        coords_y, coords_x = torch.where(mask)
        coords_t = frame_idx * torch.ones_like(coords_x)

        coordinates.append(torch.stack((coords_x, coords_y, coords_t)))

    return torch.cat(coordinates, dim=1)


def bilinear_interpolate(image: np.array, x: np.array, y: np.array):
    """ Bilinear Interpolation Function

    Interpolate Pixel Colors from a given Image

    :param image: np.array: Input Image
    :param x: np.array: X-Axis Coords Array
    :param y: np.array: Y-Axis Coords Array

    :return: np.array: Interpolated Matrix
    """

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


def stacked_run(inputs: Iterable, function: callable):
    """ Returns the result of a function for each item in a list

    :param inputs: List: List with inputs
    :param function: Callable: Function to execute the inputs

    :return: List: List with function outputs for each input
    """

    return [function(input) for input in inputs]
