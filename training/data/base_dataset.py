import PIL
import numpy as np
import torch
import random
from .easy_dataset import EasyDataset
from .augmentation import get_image_augmentation
from .dataset_util import (
    crop_image_depthmap,
    rescale_image_depthmap,
    camera_matrix_of_crop,
    bbox_from_intrinsics_in_out,
)


class BaseDataset(EasyDataset):
    """
    Define all basic options.
    """

    def __init__(
        self,
        *,  # only keyword arguments
        height=None,
        width=None,
        num_views=None,
        min_num_context_views=None,
        max_num_context_views=None,
        min_interval=1,
        max_interval=1,
        split=None,
        aug_color_jitter=False,
        aug_gray_scale=False,
        aug_gau_blur=False,
        aug_crop=False,
        aug_reverse=0.0,
        seed=None,
        seq_aug_crop=False,
        use_tqdm=False,
    ):
        assert num_views is not None, "undefined num_views"
        self.height = height
        self.width = width
        self.num_views = num_views
        self.min_num_context_views = min_num_context_views
        self.max_num_context_views = max_num_context_views if max_num_context_views is not None else min_num_context_views
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.split = split
        self.use_tqdm = use_tqdm

        # get_image_augmentation includes color jitter
        self.transform = get_image_augmentation(
            color_jitter=aug_color_jitter,
            gray_scale=aug_gray_scale,
            gau_blur=aug_gau_blur,
            img_norm=False,
        )

        self.aug_crop = aug_crop
        self.seed = seed
        self.seq_aug_crop = seq_aug_crop
        self.aug_reverse = aug_reverse

    def __len__(self):
        return len(self.scenes)

    def sample_from_video(self, video_length, num_views, min_interval, max_interval, rng, start=None, reverse=None):
        remaining_length = video_length if start is None else video_length - start
        sample_interval = np.clip(remaining_length // (num_views - 1), min_interval, max_interval)
        clip_length = (num_views - 1) * sample_interval + 1
        if start is None:
            start = rng.integers(0, max(video_length - clip_length, 0) + 1)
        end = min(start + clip_length, video_length) - 1
        sample_index = np.linspace(start, end, num_views, dtype=int)
        if reverse is None:
            reverse = rng.random() < self.aug_reverse
        if reverse:
            sample_index = sample_index[::-1]
        return sample_index, reverse

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, rng, num_context_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        num_context_views = self._rng.integers(self.min_num_context_views, self.max_num_context_views + 1)
        while True:
            try:
                views = self._get_views(idx, self._rng, num_context_views)
                break

            except Exception as e:
                print(f"Error in getting sample {idx}: {e}.")
                idx = random.randint(0, len(self) - 1)

        images = torch.from_numpy(np.stack([view["img"] for view in views]))
        images = images.permute(0, 3, 1, 2).div(255)
        images = self.transform(images) if self.transform is not None else images
        for v in range(len(views)):
            view = views[v]
            view["idx"] = (idx, v)

            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = images[v]

            if v > 0 and view["is_target"] == views[v-1]["is_target"]:
                assert view["timestamp"] >= views[v-1]["timestamp"], f"Timestamp for view {view_name(view)} is not greater than or equal to previous view."
            else:
                assert "prompt" in view

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views

    def _crop_resize_if_necessary(
        self, image, resolution, rng, info, depthmap=None, intrinsics=None
    ):
        """This function:
        - first downsizes the image with LANCZOS interpolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        if intrinsics is not None:
            cx, cy = intrinsics[:2, 2].round().astype(int)
            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)
            assert min_margin_x > W / 5, f"Bad principal point in view={info}"
            assert min_margin_y > H / 5, f"Bad principal point in view={info}"
            # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
            l, t = cx - min_margin_x, cy - min_margin_y
            r, b = cx + min_margin_x, cy + min_margin_y
            crop_bbox = (l, t, r, b)
            image, depthmap, intrinsics = crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        W, H = image.size
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        elif 0 < self.aug_crop < 1:
            delta_target_ratio = rng.random() * (1. / self.aug_crop - 1.)
            delta_target_resolution = (np.array(resolution) * delta_target_ratio).astype("int")
            target_resolution += (
                delta_target_resolution
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )

        image, depthmap, intrinsics = rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        if intrinsics is None:
            l, t = np.int32(np.round((np.array(image.size) - resolution) / 2))
            out_width, out_height = resolution
            crop_bbox = (l, t, l + out_width, t + out_height)
        else:
            # actual cropping (if necessary) with bilinear interpolation
            intrinsics2 = camera_matrix_of_crop(
                intrinsics, image.size, resolution, offset_factor=0.5
            )
            crop_bbox = bbox_from_intrinsics_in_out(
                intrinsics, intrinsics2, resolution
            )
        image, depthmap, intrinsics2 = crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )
        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    video_name = sel(view["video_name"])
    image_name = sel(view["image_name"])
    return f"{db}/{video_name}/{image_name}"
