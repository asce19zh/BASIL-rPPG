import json
import torch
import logging
import numpy as np

from .path import *
from .hdf5 import *
from .augment import *
from typing import Union
from einops import rearrange
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from .metric import predict_heart_rate_segment
import torch.nn.functional as F

@dataclass
class DatasetConfig:
    size: int
    length: int          # Actual length used for training
    raw_length: int      # Length of the loaded raw sequence, e.g., 120
    sample: int
    ratio: float
    preload: bool
    fixed_sample: bool
    augmentation: dict
    compress_factor: Union[float, list, None] = None   # Frequency compression factor


@dataclass
class LoaderConfig:
    batch_size: Union[int, list]
    num_workers: int
    shuffle: bool
    pin_memory: bool

log = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, videos, config):

        self.config = config
        self.HDF5Handler = Hdf5Handler()

        self.videos = self.HDF5Handler(videos, self.config.preload)
        self._generate_index_map()
        log.info(f'Dataset initialized with {len(self.videos)} videos, {len(self.index_map)} clips.')


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        video_key, start, end = self.index_map[idx]
        video_info = self.videos[video_key]

        if 'data' not in video_info:
            data, _ = self.HDF5Handler._preload_hdf5(video_info['path'], video_info.get('start', None), video_info.get('end', None))
        else:
            data = video_info['data']

        v_data = dict()

        if 'rgb' in data:
            v_data['entire_rgb'] = data['rgb']          # (T,H,W,C) still tensor
            raw_frames = data['rgb'][start:end]         # tensor (raw_length,H,W,C)
            
            # -----------------------------------------------------
            # 1. Original version (take the first 'length' frames directly)
            # -----------------------------------------------------
            orig = raw_frames[:self.config.length]      # (60,H,W,C)
            v_data['rgb'] = orig

            # -----------------------------------------------------
            # 2. Augmented version (raw_length -> compress -> 60)
            # -----------------------------------------------------
            if self.config.compress_factor is not None:
                cf = self._get_freq_factor()
                aug = self._apply_temporal_compress_tensor(raw_frames, cf)   # (60,H,W,C)
                v_data['rgb_aug'] = aug


        # if 'rgb_bg' in data:
        #     v_data['entire_rgb_bg'] = data['rgb_bg']
        #     v_data['rgb_bg'] = data['rgb_bg'][start:end]
        # if 'rgb_bg' not in data and 'rgb' not in data:
        #     print(f"Warning: rgb_bg not found in video {video_key}.")
        # if 'nir' in data:
        #     v_data['entire_nir'] = data['nir']
        #     v_data['nir'] = data['nir'][start:end]
        if 'gt' in data:
            gt_raw = data['gt'][start:end]                # (raw_length,)
            v_data['gt'] = gt_raw[:self.config.length]    # anchor GT

            if self.config.compress_factor is not None:
                gt_aug = self._apply_temporal_compress_gt(gt_raw, cf)
                # === HR validity check (Critical) ===
                hr_aug = predict_heart_rate_segment(
                    gt_aug,
                    Fs=30,
                    min_hr=40,
                    max_hr=180
                )
                # If out of range, do not augment
                if hr_aug is None:
                    v_data['gt_aug'] = gt_raw[:self.config.length]
                    v_data['rgb_aug'] = orig

                else:
                    v_data['gt_aug'] = gt_aug
                    
        v_data = apply_augmentation(v_data, self.config.augmentation)

        ## Resize and change channel to (C, T, H, W)
        for k in v_data.keys():

            if 'entire' in k or 'gt' in k:
                continue

            if len(v_data[k].shape) < 4:
                continue

            tensor = v_data[k].detach().clone()
            tensor = rearrange(tensor, 't h w c -> c t h w')
            tensor = tensor.float().div_(255.)
            v_data[k] = torch.nn.functional.interpolate(tensor, size=self.config.size, mode='bilinear', align_corners=False)

        ## Delete entire clip to save memory
        pop_list = list()
        for k in v_data.keys():
            if 'entire' in k:
                pop_list.append(k)

        for p in pop_list:
            v_data.pop(p)

        ## ID info for evaluation
        if len(video_key.split(',')) == 2:
            v_data['dataset'], v_data['video'] = video_key.split(',')
            v_data['subfolder'] = ''
        else:
            v_data['dataset'], v_data['subfolder'], v_data['video'] = video_key.split(',')
        v_data['clip'] = f'{start}-{end}'
        v_data['key'] = video_key

        return v_data


    def _generate_index_map(self):

        def _filter_flat_signals(gt, start, end, threshold=0.01):

            if gt is None:
                return True

            gt_segment = gt[start:end]
            diff = np.diff(gt_segment)

            # Check for 5 consecutive frames with minimal change
            for i in range(len(diff) - 5):
                if all(abs(diff[i:i+5]) < threshold):
                    return True

            return False


        self.index_map = list()

        for video_key, video_info in self.videos.items():

            # index map
            index_map = list()

            # entire video
            v_length = video_info['length']

            # get gt from video
            if self.config.preload:
                gt = video_info['data'].get('gt', None)
            else:
                gt = self.HDF5Handler.get_gt()

            # specific clip
            if 'segments' in video_info and len(video_info['segments']) > 0:
                # ori for seg in video_info['segments']:
                #    self.index_map.append((video_key, seg['start'], seg['end']))
                # continue
                v_len = video_info['length']
                for seg in video_info['segments']:
                    start = seg['start']
                    raw_end = start + self.config.raw_length

                    # Boundary protection
                    if raw_end > v_len:
                        continue

                    self.index_map.append((video_key, start, raw_end))
                continue
                
            if video_info['start'] is not None and video_info['end'] is not None:
                # ori self.index_map.append((video_key, video_info['start'], video_info['end']))
                # ori continue
                start = video_info['start']
                T = video_info['end'] - video_info['start']          # = 60
                raw_end = start + self.config.raw_length             # = start + 120

                # Boundary protection
                v_len = video_info['length']
                if raw_end > v_len:
                    continue

                self.index_map.append((video_key, start, raw_end))
                continue

            # Use raw_length as slice unit
            slice_len = self.config.raw_length

            s_num = int(v_length // slice_len * self.config.ratio)
            '''
            Test Data Augmentation

            # determine number of samples
            s_num = int(v_length // self.config.length * self.config.ratio)
            '''
            if s_num <= 0:
                s_num = int(v_length // self.config.raw_length) # wei
            if s_num <= 0:
                continue
            s_length = v_length // s_num

            # calculate each sample start and end index (key, start, end)
            for start in range(0, v_length-s_length+1, s_length):

                if self.config.fixed_sample:
                    if not _filter_flat_signals(gt, start, start + self.config.raw_length):
                        index_map.append((video_key, start, start + self.config.raw_length))
                else:
                    offset = torch.randint(0, s_length - self.config.raw_length, (1,)).item()
                    if not _filter_flat_signals(gt, start + offset, start + offset + self.config.raw_length):
                        index_map.append((video_key, start + offset, start + offset + self.config.raw_length))
                
            if self.config.sample is not None:
                if len(index_map) > self.config.sample:
                    indices = np.random.choice(len(index_map), self.config.sample, replace=False)
                    index_map = [index_map[i] for i in indices]

            self.index_map.extend(index_map)
    
    def _apply_temporal_compress_tensor(self, frames, cf):

        T = frames.shape[0]
        T_new = max(1, int(round(T / cf)))

        # Correct order: (N, C, T, H, W)
        frames = frames.float().permute(3, 0, 1, 2).unsqueeze(0)
        # (1, 3, T, H, W)

        compressed = F.interpolate(
            frames,
            size=(T_new, frames.shape[3], frames.shape[4]),
            mode='trilinear',
            align_corners=False
        )
        # (1, 3, T_new, H, W)

        final = F.interpolate(
            compressed,
            size=(self.config.length, frames.shape[3], frames.shape[4]),
            mode='trilinear',
            align_corners=False
        )
        # (1, 3, length, H, W)

        return final[0].permute(1, 2, 3, 0)
        # (length, H, W, 3)
            
    # Augment gt signal
    def _apply_temporal_compress_gt(self, gt, cf):
        """
        gt: Tensor (T,) or (T, 1)
        cf: float, frequency compression factor
        return: Tensor (length,)
        """

        if gt is None:
            return None

        if gt.dim() == 1:
            gt = gt.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif gt.dim() == 2:
            gt = gt.unsqueeze(0)                # (1, 1, T)

        T = gt.shape[-1]
        T_new = max(1, int(round(T / cf)))

        # step 1: compress / stretch
        compressed = F.interpolate(
            gt,
            size=T_new,
            mode="linear",
            align_corners=False
        )  # (1, 1, T_new)

        # step 2: resize back to fixed length
        final = F.interpolate(
            compressed,
            size=self.config.length,
            mode="linear",
            align_corners=False
        )  # (1, 1, length)

        return final[0, 0]  # (length,)



    def _get_freq_factor(self):
        f = self.config.compress_factor

        if f is None:
            return 1.0   # ⭐ No augmentation -> No compression

        if isinstance(f, (float, int)):
            return float(f)

        if isinstance(f, (list, tuple)) and len(f) == 2:
            return np.random.uniform(f[0], f[1])

        return 1.0

class ScheduledDataLoader:
    """
    A scheduled loader that automatically switches batch sizes based on the epoch.
    It behaves like an iterator, yielding a corresponding DataLoader instance for each epoch.
    """

    def __init__(self, dataset: Dataset, schedule: list, loader_config: 'LoaderConfig', total_epochs: int):
        self.dataset = dataset
        self.schedule = sorted(schedule, key=lambda x: x['end_epoch'])  # Ensure schedule is sorted by epoch
        self.loader_config = loader_config
        self.total_epochs = total_epochs
        self._loaders = {}  # Cache for created DataLoaders

    def __iter__(self):
        """
        Allows this class to be iterated in a for loop.
        It yields the correct DataLoader at each epoch.
        """
        current_schedule_idx = 0
        for epoch in range(self.total_epochs):
            # Check if we need to switch to the next schedule
            if epoch >= self.schedule[current_schedule_idx]['end_epoch']:
                if current_schedule_idx < len(self.schedule) - 1:
                    current_schedule_idx += 1

            batch_size = self.schedule[current_schedule_idx]['batch_size']

            # If loader for batch_size is not created, create one
            if batch_size not in self._loaders:
                self._loaders[batch_size] = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=self.loader_config.shuffle,
                    num_workers=self.loader_config.num_workers,
                    pin_memory=self.loader_config.pin_memory
                )

            # Yield the loader corresponding to the current epoch
            yield self._loaders[batch_size]


# def get_loader(protocols, dataset_config, loader_config, total_epochs=None):
#     data = list()
#     log.info('Using protocols: ' + " ,".join(protocols))

#     for i, (_json, _txt) in enumerate(pathManager.get_protocol_path(protocols)):
#         if not (_json.exists() or _txt.exists()):
#             raise Exception(f"Protocol {protocols[i]} is not available")
#         if not _json.exists():
#             _transfer_protcol(_txt, _json)
#         with open(_json, 'r') as f:
#             videos = json.load(f)
#         data.extend(videos)

#     data = list(tuple(data))
#     log.info('Number of videos: ' + str(len(data)))

#     dataset = VideoDataset(data, dataset_config)

#     # If batch_size is not a list, it represents a single, fixed batch_size
#     if not isinstance(loader_config.batch_size, list):
#         loader = DataLoader(dataset, batch_size=loader_config.batch_size, shuffle=loader_config.shuffle,
#                             num_workers=loader_config.num_workers, pin_memory=loader_config.pin_memory)
#         # To unify the interface with the scheduled loader, return a list containing a single loader
#         return [loader] * (total_epochs or 1)

#     # If batch_size is a list, scheduling is required
#     else:
#         if total_epochs is None:
#             raise ValueError("total_epochs must be provided when using scheduled batch_size.")
#         # Return scheduled loader instance
#         return ScheduledDataLoader(dataset, loader_config.batch_size, loader_config, total_epochs)

# def _transfer_protcol(txt_path, json_path):

#     videos = list()
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.strip() == "" or line.startswith("#"):
#                 continue
#             parts = line.strip().split(',')
#             if len(parts) == 2:
#                 videos.append({'dataset': parts[0], 'video': parts[1]})
#             elif len(parts) == 3:
#                 videos.append({'dataset': parts[0], 'subfolder': parts[1], 'video': parts[2]})
#             else:
#                 raise Exception(f"Invalid protocol line: {line.strip()}")

#     sorted_videos = sorted(videos, key=lambda x: x['dataset']+x['video'] if 'subfolder' not in x else x['dataset']+x['subfolder']+x['video'])

#     with open(json_path, 'w') as f:
#         json.dump(sorted_videos, f, indent=4)
def get_loader(protocols, dataset_config, loader_config, total_epochs=None):
    data = list()
    log.info('Using protocols: ' + " ,".join(protocols))

    for i, (_json, _txt) in enumerate(pathManager.get_protocol_path(protocols)):

        if not (_json.exists() or _txt.exists()):
            raise Exception(f"Protocol {protocols[i]} is not available")
        
        # Check if JSON file exists in the local protocol directory
        import os
        local_protocol_dir = os.path.join(os.path.dirname(__file__), '..', 'protocol')
        local_json_path = os.path.join(local_protocol_dir, os.path.basename(_json))
        
        # Prioritize local JSON; convert from TXT if it doesn't exist
        if os.path.exists(local_json_path):
            with open(local_json_path, 'r') as f:
                videos = json.load(f)
        elif _json.exists():
            with open(_json, 'r') as f:
                videos = json.load(f)
        else:
            _transfer_protcol(_txt, _json)
            with open(local_json_path, 'r') as f:
                videos = json.load(f)
        
        data.extend(videos)
    data = list(tuple(data))
    log.info('Number of videos: ' + str(len(data)))
    dataset = VideoDataset(data, dataset_config)

    # If batch_size is not a list, it represents a single, fixed batch_size
    if not isinstance(loader_config.batch_size, list):
        loader = DataLoader(dataset, batch_size=loader_config.batch_size, shuffle=loader_config.shuffle,
                            num_workers=loader_config.num_workers, pin_memory=loader_config.pin_memory)
        # To unify the interface with the scheduled loader, return a list containing a single loader
        return [loader] * (total_epochs or 1)

    # If batch_size is a list, scheduling is required
    else:
        if total_epochs is None:
            raise ValueError("total_epochs must be provided when using scheduled batch_size.")
        # Return scheduled loader instance
        return ScheduledDataLoader(dataset, loader_config.batch_size, loader_config, total_epochs)
def _transfer_protcol(txt_path, json_path):

    videos = list()
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split(',')
            if len(parts) == 2:
                videos.append({'dataset': parts[0], 'video': parts[1]})
            elif len(parts) == 3:
                videos.append({'dataset': parts[0], 'subfolder': parts[1], 'video': parts[2]})
            else:
                raise Exception(f"Invalid protocol line: {line.strip()}")

    sorted_videos = sorted(videos, key=lambda x: x['dataset']+x['video'] if 'subfolder' not in x else x['dataset']+x['subfolder']+x['video'])

    # Save the JSON file to the project's local protocol directory
    import os
    local_protocol_dir = os.path.join(os.path.dirname(__file__), '..', 'protocol')
    os.makedirs(local_protocol_dir, exist_ok=True)
    local_json_path = os.path.join(local_protocol_dir, os.path.basename(json_path))
    
    with open(local_json_path, 'w') as f:
        json.dump(sorted_videos, f, indent=4)
    
    log.info(f"Protocol converted and saved to {local_json_path}")

def reset_seed():
    reset_testing_seed()