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

@dataclass
class DatasetConfig:
    size: int
    length: int
    sample: int
    ratio: float
    preload: bool
    fixed_sample: bool
    augmentation: dict


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
            v_data['entire_rgb'] = data['rgb']
            v_data['rgb'] = data['rgb'][start:end]
        if 'rgb_bg' in data:
            v_data['entire_rgb_bg'] = data['rgb_bg']
            v_data['rgb_bg'] = data['rgb_bg'][start:end]
        if 'nir' in data:
            v_data['entire_nir'] = data['nir']
            v_data['nir'] = data['nir'][start:end]
        if 'gt' in data:
            v_data['gt'] = data['gt'][start:end]

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
                for seg in video_info['segments']:
                    self.index_map.append((video_key, seg['start'], seg['end']))
                continue

            if video_info['start'] is not None and video_info['end'] is not None:
                self.index_map.append((video_key, video_info['start'], video_info['end']))
                continue

            # determine number of samples
            log.info(v_length, ", length = ", self.config.length, ", ratio = ", self.config.ratio)
            s_num = int(v_length // self.config.length * self.config.ratio)
            if s_num <= 0:
                s_num = int(v_length // self.config.length)
            if s_num <= 0:
                continue
            s_length = v_length // s_num

            # calculate each sample start and end index (key, start, end)
            for start in range(0, v_length-s_length+1, s_length):

                if self.config.fixed_sample:
                    if not _filter_flat_signals(gt, start, start + self.config.length):
                        index_map.append((video_key, start, start + self.config.length))
                else:
                    offset = torch.randint(0, s_length - self.config.length, (1,)).item()
                    if not _filter_flat_signals(gt, start + offset, start + offset + self.config.length):
                        index_map.append((video_key, start + offset, start + offset + self.config.length))

            if self.config.sample is not None:
                if len(index_map) > self.config.sample:
                    indices = np.random.choice(len(index_map), self.config.sample, replace=False)
                    index_map = [index_map[i] for i in indices]

            self.index_map.extend(index_map)


class ScheduledDataLoader:
    """
    一個排程載入器，可以根據 epoch 自動切換不同的 batch size。
    它的行為類似一個迭代器，每個 epoch 會 yield 一個對應的 DataLoader 實例。
    """

    def __init__(self, dataset: Dataset, schedule: list, loader_config: 'LoaderConfig', total_epochs: int):
        self.dataset = dataset
        self.schedule = sorted(schedule, key=lambda x: x['end_epoch'])  # 確保排程按 epoch 排序
        self.loader_config = loader_config
        self.total_epochs = total_epochs
        self._loaders = {}  # 用於快取已建立的 DataLoader

    def __iter__(self):
        """
        讓這個類別可以被 for 迴圈迭代。
        在每個 epoch，它會 yield 正確的 DataLoader。
        """
        current_schedule_idx = 0
        for epoch in range(self.total_epochs):
            # 檢查是否需要切換到下一個排程
            if epoch >= self.schedule[current_schedule_idx]['end_epoch']:
                if current_schedule_idx < len(self.schedule) - 1:
                    current_schedule_idx += 1

            batch_size = self.schedule[current_schedule_idx]['batch_size']

            # 如果對應 batch_size 的 loader 還沒建立，就建立一個
            if batch_size not in self._loaders:
                self._loaders[batch_size] = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=self.loader_config.shuffle,
                    num_workers=self.loader_config.num_workers,
                    pin_memory=self.loader_config.pin_memory
                )

            # Yield 當前 epoch 對應的 loader
            yield self._loaders[batch_size]


def get_loader(protocols, dataset_config, loader_config, total_epochs=None):
    data = list()
    log.info('Using protocols: ' + " ,".join(protocols))

    for i, (_json, _txt) in enumerate(pathManager.get_protocol_path(protocols)):
        if not (_json.exists() or _txt.exists()):
            raise Exception(f"Protocol {protocols[i]} is not available")
        if not _json.exists():
            _transfer_protcol(_txt, _json)
        with open(_json, 'r') as f:
            videos = json.load(f)
        data.extend(videos)

    new_data = list()
    for v_info in data:
        if 'source1' not in v_info['video']:
            new_data.append(v_info)

    data = list(tuple(new_data))
    log.info('Number of videos: ' + str(len(data)))

    dataset = VideoDataset(data, dataset_config)

    # 如果 batch_size 不是 list，代表是單一、固定的 batch_size
    if not isinstance(loader_config.batch_size, list):
        loader = DataLoader(dataset, batch_size=loader_config.batch_size, shuffle=loader_config.shuffle,
                            num_workers=loader_config.num_workers, pin_memory=loader_config.pin_memory)
        # 為了與排程載入器介面統一，我們回傳一個只包含單一 loader 的 list
        return [loader] * (total_epochs or 1)

    # 如果 batch_size 是 list，代表需要排程
    else:
        if total_epochs is None:
            raise ValueError("使用排程 batch_size 時必須提供 total_epochs。")
        # 回傳排程載入器實例
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
            elif len(parts) == 1:
                videos.append({'dataset': 'COHFACE', 'video': parts[0]})
            else:
                raise Exception(f"Invalid protocol line: {line.strip()}")

    sorted_videos = sorted(videos, key=lambda x: x['dataset']+x['video'] if 'subfolder' not in x else x['dataset']+x['subfolder']+x['video'])

    with open(json_path, 'w') as f:
        json.dump(sorted_videos, f, indent=4)

def reset_seed():
    reset_testing_seed()