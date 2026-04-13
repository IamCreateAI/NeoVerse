import os.path as osp
import numpy as np
import cv2
import numpy as np
import json
import os
import sys
import pandas as pd
from decord import VideoReader
import gc
from contextlib import contextmanager

from tqdm import tqdm
from ..base_dataset import BaseDataset


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


class SpatialVID(BaseDataset):
    def __init__(self, ROOT, *args, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        metadata = pd.read_csv(osp.join(self.ROOT, "data/train/SpatialVID_HQ_metadata.csv"))
        min_anno_length = (self.num_views - 1) * self.min_interval + 1
        annotation_interval = (0.2 * metadata["fps"]).astype(int)
        min_clip_length = annotation_interval * (min_anno_length - 1) + 1
        self.scenes = metadata[metadata["num frames"] >= min_clip_length]

    def __len__(self):
        return len(self.scenes)

    def _get_views(self, idx, rng, num_context_views):
        scene_info = self.scenes.iloc[idx]
        video_path = osp.join(self.ROOT, "SpatialVid/HQ", scene_info["video path"])
        annotation_dir = osp.join(self.ROOT, "SpatialVid/HQ", scene_info["annotation path"])

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            video_length = len(video_reader)
            sample_index, reverse = self.sample_from_video(
                video_length, self.num_views, self.min_interval, self.max_interval, rng
            )
            sample_context_index = sample_index[np.linspace(0, self.num_views - 1, num_context_views, dtype=int)]
            images = video_reader.get_batch(sample_index).asnumpy()

        with open(osp.join(annotation_dir, "caption.json"), 'r') as f:
            captions = json.load(f)
            text_prompt = captions["SceneDescription"]

        context_views = []
        target_views = []
        for v, rgb_image in enumerate(images):
            timestamp = sample_index[v] - sample_index[0] if not reverse else sample_index[0] - sample_index[v]
            rgb_image, *_ = self._crop_resize_if_necessary(
                rgb_image, (self.width, self.height), rng=rng, info=(idx, v),
            )
            if sample_index[v] in sample_context_index:
                context_views.append(
                    dict(
                        img=rgb_image,
                        dataset="SpatialVID",
                        video_name=scene_info["id"],
                        image_name=f"frame_{sample_index[v]:06d}",
                        is_static=False,
                        is_target=False,
                        timestamp=timestamp,
                        prompt=text_prompt,
                    )
                )
            else:
                target_views.append(
                    dict(
                        img=rgb_image,
                        dataset="SpatialVID",
                        video_name=scene_info["id"],
                        image_name=f"frame_{sample_index[v]:06d}",
                        is_static=False,
                        is_target=True,
                        timestamp=timestamp,
                        prompt=text_prompt,
                    )
                )
        views = context_views + target_views
        return views
