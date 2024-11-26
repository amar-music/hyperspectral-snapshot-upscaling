# flake8: noqa
import os.path as osp
import sys
from basicsr.train import train_pipeline

sys.path.insert(0, "/home/amarus/GitHub/hyperspectral-snapshot-upscaling/Real-ESRGAN/")

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
