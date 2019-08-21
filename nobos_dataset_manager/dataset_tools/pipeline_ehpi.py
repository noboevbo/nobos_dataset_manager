from typing import List, Dict

import numpy as np
import torch
from ehpi_action_recognition import config
from ehpi_action_recognition.networks.pose_estimation_2d_nets.pose2d_net_resnet import Pose2DNetResnet
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.human import Human
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.data_structures.image_content_buffer import ImageContentBuffer
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.input_providers.input_provider_base import InputProviderBase
from nobos_commons.tools.log_handler import logger
from nobos_commons.tools.pose_tracker import PoseTracker
from nobos_torch_lib.models.pose_estimation_2d_models import pose_resnet

from nobos_dataset_manager import configurator


class PipelineEhpi:
    def __init__(self, image_size: ImageSize, use_pose_tracking: bool, use_fast_mode: bool):
        configurator.setup()
        self.__current_worker_id = None
        self.buffer_size = 20
        # Settings
        self.skeleton_type = SkeletonStickman
        self.use_fast_mode = use_fast_mode
        self.image_size = image_size

        self.pose_recognition = True
        self.pose_tracking = use_pose_tracking

        # Pose Network
        self.pose_model = pose_resnet.get_pose_net(config.pose_resnet_config)
        self.action_names = [Action.IDLE.name, Action.WALK.name, Action.WAVE.name]

        logger.info('=> loading model from {}'.format(config.pose_resnet_config.model_state_file))
        self.pose_model.load_state_dict(torch.load(config.pose_resnet_config.model_state_file))
        self.pose_model = self.pose_model.cuda()
        self.pose_model.eval()
        self.pose_net = Pose2DNetResnet(self.pose_model, self.skeleton_type)
        self.pose_tracker = PoseTracker(image_size=self.image_size, skeleton_type=self.skeleton_type)

        self.image_content_buffer: ImageContentBuffer = None

    def redetect_with_fastmode(self, last_humans: List[Human]):
        return not self.use_fast_mode or last_humans is None or len(last_humans) == 0

    def get_from_input_provider(self, input_provider: InputProviderBase):
        frame_results: Dict[int, ImageContent] = {}
        for frame_nr, frame in enumerate(input_provider.get_data()):
            image_content = self.get_image_content(frame)
            if self.image_content_buffer is not None:
                self.image_content_buffer.add(image_content)
            frame_results[frame_nr] = image_content
        return frame_results

    def get_image_content(self, frame: np.ndarray):
        humans = []
        object_bounding_boxes = []
        if self.pose_recognition:
            if self.pose_tracking:
                if self.image_content_buffer is None:
                    self.image_content_buffer = ImageContentBuffer(buffer_size=self.buffer_size)
                last_humans = self.image_content_buffer.get_last_humans()
                if self.redetect_with_fastmode(last_humans):
                    object_bounding_boxes = self.pose_net.detector.get_object_bounding_boxes(frame)
                    human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                    humans = self.pose_net.get_humans_from_bbs(frame, human_bbs)
                humans, undetected_humans = self.pose_tracker.get_humans_by_tracking(frame, detected_humans=humans,
                                                                                     previous_humans=last_humans)
                redetected_humans = self.pose_net.redetect_humans(frame, undetected_humans, min_human_score=0.4)
                humans.extend(redetected_humans)
            else:
                if self.image_content_buffer is not None:
                    self.image_content_buffer = None
                object_bounding_boxes = self.pose_net.detector.get_object_bounding_boxes(frame)
                human_bbs = [bb for bb in object_bounding_boxes if bb.label == "person"]
                humans = self.pose_net.get_humans_from_bbs(frame, human_bbs, min_human_score=0.4)
        else:
            object_bounding_boxes = self.pose_net.detector.get_object_bounding_boxes(frame)
        human_bbs = [human.bounding_box for human in humans]
        other_bbs = [bb for bb in object_bounding_boxes if bb.label != "person"]

        return ImageContent(humans=humans, objects=human_bbs + other_bbs)
