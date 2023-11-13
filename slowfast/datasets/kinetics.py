#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import pandas
import torch
import torch.utils.data
from torchvision import transforms
import cv2
import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import (
    MaskingGenerator,
    MaskingGenerator3D,
    create_random_augment,
)

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.

    VRC video loader. 
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the VRC video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for VRC".format(mode)
        self.mode = mode
        self.cfg = cfg

        ## to set
        
        self.IMAGE_HEIGHT= 90*3
        self.IMAGE_WIDTH= 120*3
        if "checkpoints_X3D_XS" in self.cfg.OUTPUT_DIR :
                self.IMAGE_HEIGHT= 90
                self.IMAGE_WIDTH= 120
        self.NUM_FRAMES= self.cfg.DATA.NUM_FRAMES
        self.CHANNELS= 3
        self.normalize = True
        self.use_position = False  
        self.use_newer_model = False  
        self.num_classes = 13
        if self.cfg.MODEL.ARCH == "mvit":
            self.IMAGE_HEIGHT = 224
            self.IMAGE_WIDTH = 224
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
            self.augment = True
        elif self.mode in ["test"]:
            self._num_clips = 1 # important for our work since we can only allow one clip per video
            self.augment = False
        logger.info("Constructing VRC {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        self._path_to_videos = []
        self._labels = []
        self.all_hardness = {}
        self._hardness = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS

        # self._path_to_videos = np.random.permutation(open(path_to_file).readlines())

        path_to_meta_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "meta.csv".format(self.mode)
        )
        with pathmgr.open(path_to_meta_file, "r") as f:
            meta_data = f.read().splitlines()
        for option in meta_data:
            name, method = option.split(";")
            self.all_hardness[name] = method
            
        with pathmgr.open(path_to_file, "r") as f:
            if self.use_chunk_loading:
                rows = self._get_chunk(f, self.cfg.DATA.LOADER_CHUNK_SIZE)
            else:
                rows = f.read().splitlines()
            for clip_idx, path_label in enumerate(rows):
                fetch_info = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                path = fetch_info[0]
                label = fetch_info[-1]
                if "CKG" in path and not self.cfg.DATASET_TYPE.CKG:
                    continue
                if "CKF" in path and not self.cfg.DATASET_TYPE.CKF:
                    continue
                if "TST" in path and not self.cfg.DATASET_TYPE.TST:
                    continue
                if "SYN" in path:
                    if self.all_hardness.get(path)=="easy" and not self.cfg.DATASET_TYPE.SYN_EASY:
                        continue
                    if self.all_hardness.get(path)=="hard" and not self.cfg.DATASET_TYPE.SYN_HARD:
                        continue
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                    if self.all_hardness.get(path)=="hard":
                        self._hardness.append(1)
                    else:
                        self._hardness.append(0)
                        
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load VRC split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing VRC dataloader (size: {} skip_rows {}) from {} ".format(
                len(self._path_to_videos), self.skip_rows, path_to_file
            )
        )
        
        
    
    def flip_label(self, label):
        if label not in range(13):
            # incorrect label
            return None
        elif label in [0, 11, 12]:
            # symmetric gestures
            return label
        
        # switch even and odd labels
        if label % 2 == 0:
            return label-1
        else:
            return label+1

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            for chunk in pandas.read_csv(
                path_to_file,
                chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                skiprows=self.skip_rows,
            ):
                break
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")

    def __getitem__(self, index):
        folder_path = self._path_to_videos[index]
        label = self._labels[index]
        imgs = [name for name in os.listdir(folder_path)]
        imgs = sorted(imgs) #Need to sort
        # issues with loading via cfg
        data = np.zeros((self.NUM_FRAMES, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        if self.mode in ["train", "val"]:
            flip = np.random.randint(2) # flip this entire sequence
            if flip:
                label = self.flip_label(label) 

        if self.augment:
            # randomize values for contrast and brightness
            alpha = np.random.uniform(0.75, 1.25)
            beta = np.random.uniform(-15,15)
        else:
            alpha, beta = 1, 0
        for idx in range(self.NUM_FRAMES): # take frames starting from the beginning (even if less frames needed)
            if len(imgs) <= idx:
                # if we are out of frames, take the last frame
                image_path = os.path.join(folder_path, imgs[-1])
            else:
                image_path = os.path.join(folder_path, imgs[idx])
            image = cv2.imread(image_path).astype(np.float32) # idx counts from 0 upwards
            image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

            if flip:
                image = cv2.flip(image, 1)

            if self.augment:
                # small random changes in contrast and brightness
                alpha += np.random.uniform(-0.05,0.05)
                beta += np.random.uniform(-2,2)

                # random crop (not too small so don't cut of referee)
                crop_ratio = np.random.uniform(0.9, 0.99)
                crop_width = int(crop_ratio * self.IMAGE_WIDTH)
                crop_height = int(crop_ratio * self.IMAGE_HEIGHT)
                x = np.random.randint(0, self.IMAGE_WIDTH - crop_width)
                y = np.random.randint(0, self.IMAGE_HEIGHT - crop_height)
                image = image[y:y+crop_height, x:x+crop_width]
                image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            if self.normalize:
                scale  = 2
                add = -1
            else:
                scale = 1
                add = 0
            data[idx, :, :, 0] = image[:, :, 0] / 255*scale + add
            data[idx, :, :, 1] = image[:, :, 1] / 255*scale + add
            data[idx, :, :, 2] = image[:, :, 2] / 255*scale + add
        #transform data dimension from (frames, height, width, channels) to (channels, frames, height, width)
        if self.cfg.MODEL.ARCH != "mvit":
            data = torch.tensor(np.transpose(data, (3, 0, 1, 2))).float()
        else:
            data = torch.tensor(np.transpose(data, (3, 0, 1, 2))).float()
        frames = utils.pack_pathway_output(self.cfg, data)
        time = 0
        if self.cfg.MODEL.ARCH == "mvit":
            frames = [frames]
            label = [label]
            index = [index]
            time = [time]
        return frames, label, index, time, {}
        # How the return should look like:
        return frames, label, index, time_idx, {}


    def get_slow_data(self, data, skip=2):
        slow_data = np.zeros((self.CHANNELS, self.NUM_FRAMES//skip, self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        for i in range(self.NUM_FRAMES//skip):
            slow_data[:, i, :, :] = data[:, i*skip, :, :]
        return slow_data

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=int)
            n_mask = round(
                self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
