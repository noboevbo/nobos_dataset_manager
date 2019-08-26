import os

from ehpi_action_recognition import config
from nobos_dataset_api.configs import config as api_cfg
from ehpi_action_recognition import config as ehpi_cfg
import torch

curr_dir = os.path.dirname(os.path.realpath(__file__))


def setup():
    # Torch Backend Settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    api_cfg.reload_all = True
    ehpi_cfg.cache_config.reload_all = True
    a = api_cfg

    config.cache_config.cache_dir = "/media/disks/beta/app_data/cache/ofp_ui"
    config.yolo_v3_config.model_state_file = "/media/disks/beta/app_data/ehpi_action_recognition/data/models/yolov3.weights"
    config.yolo_v3_config.network_config_file = "/media/disks/beta/app_data/ehpi_action_recognition/data/configs/yolo_v3.cfg"
    config.pose_resnet_config.model_state_file = "/media/disks/beta/app_data/ehpi_action_recognition/data/models/pose_resnet_50_256x192.pth.tar"
    config.ehpi_model_state_file = "/media/disks/beta/app_data/ehpi_action_recognition/data/models/ehpi_v1.pth"
