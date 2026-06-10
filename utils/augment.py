import os
from .augmentaion import *


def reset_testing_seed():
    pass

def apply_augmentation(video, aug):

    enable = aug.get('enable', False)
    name = aug.get('name', None)

    if not enable:
        return video

    if name is None or name == 'None':
        name = os.environ.get('model')

    return augmentation(video, name)