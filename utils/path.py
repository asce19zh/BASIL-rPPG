import os
import time

from pathlib import Path
from dotenv import load_dotenv

MAX_FILE_NUM = 50

class PathManager:

    def __init__(self):
        load_dotenv('.env')

        if os.environ.get("DIR") is None:
            raise Exception("Please set the DIR environment variable in .env file")

        self.dataset = dict()
        self.dir = Path(os.environ.get("DIR"))

        if (self.dir / "raw").exists():
            self.dataset.update({d.stem.lower(): d.stem for d in (self.dir / "raw").iterdir() if d.is_dir()})
        if (self.dir / "cropped").exists():
            self.dataset.update({d.stem.lower(): d.stem for d in (self.dir / "cropped").iterdir() if d.is_dir()})
        if (self.dir / "hdf5").exists():
            self.dataset.update({d.stem.lower(): d.stem for d in (self.dir / "hdf5").iterdir() if d.is_dir()})

    ## TODO: Add raw path getter
    def get_raw_path(self):
        pass

    def get_cropped_path(self, dataset, modality, subfolder=None, video_name=None):
        if dataset.lower() not in self.dataset.keys():
            raise Exception(f"Dataset {dataset} is not available")
        if video_name is None:
            raise Exception("Please provide video_name")
        if subfolder is None:
            return self.dir / "cropped" / self.dataset[dataset.lower()] / modality / video_name
        return self.dir / "cropped" / self.dataset[dataset.lower()] / modality / subfolder / video_name

    def get_hdf5_path(self, dataset, subfolder, video_name):
        if dataset.lower() not in self.dataset.keys():
            raise Exception(f"Dataset {dataset} is not available")
        self.make_hdf5_dir(self.dataset[dataset.lower()], subfolder)
        if subfolder is None:
            return self.dir / "hdf5" / self.dataset[dataset.lower()] / f"{video_name}.h5"
        return self.dir / "hdf5" / self.dataset[dataset.lower()] / subfolder /f"{video_name}.h5"

    def get_protocol_path(self, protocols):
        for p in protocols:
            yield (self.dir / "protocol" / f"{p}.json", self.dir / "protocol" / f"{p}.txt")

    def get_weight_path(self, model, protocol, length, epoch, component=None):
        protocol = ",".join(protocol) if isinstance(protocol, list) else protocol
        base_path = self.dir / "weight" / model / protocol.lower() / f"length_{length:03d}"
        if component is None:
            self.make_dir(base_path)
            return base_path / f"epoch_{epoch:04d}.pth"

        component_path = base_path / component
        self.make_dir(component_path)
        return component_path / f"{epoch:04d}.pth"

    def get_output_path(self, model, train_protocol=[], inference_protocol=[], vid=None, epoch=None):
        train_protocol = ",".join(train_protocol) if isinstance(train_protocol, list) else train_protocol
        inference_protocol = ",".join(inference_protocol) if isinstance(inference_protocol, list) else inference_protocol
        if not inference_protocol:
            raise Exception("Please provide inference_protocol for output path")
        name = f"{train_protocol}_to_{inference_protocol}" if train_protocol else inference_protocol
        path = self.dir / "output" / model / name.lower()

        paths = []
        for id in vid:
            paths.append(path / id / f"epoch_{epoch:04d}")
            self.make_dir(paths[-1])
        return paths


    def get_log_path(self, stage, model, train_protocol=[], inference_protocol=[]):
        train_protocol = ",".join(train_protocol) if isinstance(train_protocol, list) else train_protocol
        inference_protocol = ",".join(inference_protocol) if isinstance(inference_protocol, list) else inference_protocol
        if stage == "train":
            if not train_protocol:
                raise Exception("Please provide train_protocol for training log path")
            path = Path("./log") / stage / train_protocol.lower() / model
        else:
            if not inference_protocol:
                raise Exception("Please provide inference_protocol for inference log path")
            name = f"{train_protocol}_to_{inference_protocol}" if train_protocol else inference_protocol
            path = Path("./log") / stage / name.lower() / model

        self.make_dir(path)

        # TODO: info_{timestamp}.log and detail_{timestamp}.log && info.log and detail.log
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        info_path = path / f"info_{timestamp}.log"
        detail_path = path / f"detail_{timestamp}.log"

        # TODO: info_{timestamp}.log and detail_{timestamp}.log only ten files, the other old files will be deleted
        existed_info_logs = sorted([f for f in path.glob("info_*.log") if f.is_file()], key=os.path.getmtime)
        existed_detail_logs = sorted([f for f in path.glob("detail_*.log") if f.is_file()], key=os.path.getmtime)
        if len(existed_info_logs) >= MAX_FILE_NUM:
            for f in existed_info_logs[:len(existed_info_logs) - MAX_FILE_NUM + 1]:
                os.remove(f)
        if len(existed_detail_logs) >= MAX_FILE_NUM:
            for f in existed_detail_logs[:len(existed_detail_logs) - MAX_FILE_NUM + 1]:
                os.remove(f)

        return info_path, detail_path

    def make_hdf5_dir(self, dataset, subfolder):
        if dataset.lower() not in self.dataset.keys():
            raise Exception(f"Dataset {dataset} is not available")
        if subfolder is None:
            path = self.dir / "hdf5" / self.dataset[dataset.lower()]
        else:
            path = self.dir / "hdf5" / self.dataset[dataset.lower()] / subfolder
        self.make_dir(path)
        return path

    def make_dir(self, path):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)



pathManager = PathManager()