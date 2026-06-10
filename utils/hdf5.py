import h5py
import math
import torch
import logging
import numpy as np
import concurrent.futures

from .path import *
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision import transforms

log = logging.getLogger(__name__)

class Hdf5Handler:

    def __init__(self):
        self._size = 128
        self._transforms = transforms.Compose([
            transforms.Resize((self._size, self._size)),
        ])

    def __call__(self, videos, preload=True):

        num_workers = 6
        video_data = dict()
        
        # Group videos to avoid duplicate loading
        unique_videos = {}
        for v in videos:
            key = f"{v['dataset']},{v['subfolder']},{v['video']}" if 'subfolder' in v.keys() else f"{v['dataset']},{v['video']}"
            if key not in unique_videos:
                unique_videos[key] = {
                    'video_info': v,
                    'segments': []
                }
            if 'start' in v and 'end' in v:
                unique_videos[key]['segments'].append({'start': v['start'], 'end': v['end']})

        length = len(unique_videos)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {executor.submit(self._get_data_from_a_video, info['video_info'], preload): key for key, info in unique_videos.items()}
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_video), total=length, desc="Loading videos")):
                k, v = future.result()
                # Attach segments
                v['segments'] = unique_videos[k]['segments']
                video_data[k] = v

        return video_data

    def _get_data_from_a_video(self, v, preload=True):

        path = pathManager.get_hdf5_path(v.get('dataset'), v.get('subfolder', None), v.get('video'))
        key = f"{v['dataset']},{v['subfolder']},{v['video']}" if 'subfolder' in v.keys() else f"{v['dataset']},{v['video']}"

        # if hdf5 not found, create one
        if not path.exists():
            self._create_hdf5(v)

        # preload hdf5 data
        if preload:
            data, length = self._preload_hdf5(v)
            return key, {'data': data, 'length': length, 'start': v.get('start', None), 'end': v.get('end', None)}
        else:
            return key, {'path': path, 'length': self._get_length_from_hdf5(path), 'start': v.get('start', None), 'end': v.get('end', None)}

    def _create_hdf5(self, v):

        def _get_images_from_folder(folder):
            images = sorted([p for p in folder.iterdir() if p.suffix in ['.png', '.jpg', '.jpeg']])
            data = []
            for img_path in images:
                with Image.open(img_path) as img:
                    resized_img = self._transforms(img)
                    img_array = np.array(resized_img)
                    data.append(img_array)
            data = data[30:]
            data = np.stack(data, axis=0)
            return data

        rgb_path = pathManager.get_cropped_path(v['dataset'], 'RGB_crop', v.get('subfolder', None), v['video'])
        rgb_bg_path = pathManager.get_cropped_path(v['dataset'], 'RGB_bg', v.get('subfolder', None), v['video'])
        nir_path = pathManager.get_cropped_path(v['dataset'], 'NIR_crop', v.get('subfolder', None), v['video'])
        gt_path = pathManager.get_cropped_path(v['dataset'], 'GT', None, v['video']) / 'ground_truth.txt'

        with h5py.File(pathManager.get_hdf5_path(v['dataset'], v.get('subfolder', None), v['video']), 'w') as hdf:
            length = math.inf
            if rgb_path.exists():
                rgb_data = _get_images_from_folder(rgb_path)
                length = min(length, rgb_data.shape[0])
            if rgb_bg_path.exists():
                rgb_bg_data = _get_images_from_folder(rgb_bg_path)
                length = min(length, rgb_bg_data.shape[0])
            if nir_path.exists():
                nir_data = _get_images_from_folder(nir_path)
                length = min(length, nir_data.shape[0])
            if gt_path.exists():
                with open(gt_path, 'r') as f:
                    gt_line = f.readline().strip().split()
                    gt_data = np.array([float(x) for x in gt_line][30:], dtype=np.float32)
                length = min(length, gt_data.shape[0])

            hdf.create_dataset('length', data=length)
            if 'rgb_data' in locals():
                hdf.create_dataset('rgb', data=rgb_data[:length], compression="gzip")
            if 'rgb_bg_data' in locals():
                hdf.create_dataset('rgb_bg', data=rgb_bg_data[:length], compression="gzip")
            if 'nir_data' in locals():
                hdf.create_dataset('nir', data=nir_data[:length], compression="gzip")
            if 'gt_data' in locals():
                hdf.create_dataset('gt', data=gt_data[:length], compression="gzip")

    def _preload_hdf5(self, v):
        data = {}
        path = pathManager.get_hdf5_path(v.get('dataset'), v.get('subfolder', None), v.get('video'))
        try:
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    if key != 'length':
                        data[key] = torch.from_numpy(f[key][:])
                length = f['length'][()]
            return data, length
        except Exception as e:
            self._create_hdf5(v)
            return self._preload_hdf5(v)

    def _get_length_from_hdf5(self, v):
        path = pathManager.get_hdf5_path(v.get('dataset'), v.get('subfolder', None), v.get('video'))
        try:
            with h5py.File(path, 'r') as f:
                return f['length']
        except Exception as e:
            self._create_hdf5(v)
            return self._get_length_from_hdf5(v)

