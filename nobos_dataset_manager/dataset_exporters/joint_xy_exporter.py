import os
from typing import List, Iterator, Dict

import numpy as np
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action, jhmdb_actions
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_joint_config import \
    get_joints_jhmdb
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_simple_joints import \
    FeatureVecProducerSimpleJoints
from nobos_commons.utils.numpy_helper import split_numpy_array_stepwise, set_or_vstack

from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth
from nobos_dataset_manager.utils import get_skeleton_from_skeleton_db


class JointXyExporter(object):
    def __init__(self):
        self.skeleton = SkeletonStickman()
        self.feature_vec_producers: Dict[(int, int), FeatureVecProducerSimpleJoints] = {}

    def get_ehpi_images(self, data: Iterator[HumanAction], num_actions: int, output_path: str, dataset_part: DatasetPart):
        xs: np.ndarray = None
        ys: List[List[int]] = []
        for idx, human_action in enumerate(data):
            print("Working on action sequence ({0}/{1}): '{2}'".format(idx, num_actions, human_action.video_gt.vid_name))
            action = Action(human_action.action)
            feature_vec_producer = self.__get_feature_vec_producer(human_action)

            feature_vecs: List[np.ndarray] = []
            for human in sorted(human_action.humans,
                                key=lambda e: (e.uid, e.frame_gt.frame_num)):  # Order by human and frame_num
                db_joints = list(human.joints)[0]
                skeleton = get_skeleton_from_skeleton_db(db_joints)
                feature_vec = feature_vec_producer.get_feature_vec(skeleton)
                feature_vecs.append(np.ravel(feature_vec))
                # TODO: Should be 32xnum_jointsx2 as lstm input
            stepwise_splits = split_numpy_array_stepwise(np.array(feature_vecs), split_size=32,
                                                         step_size=1, fill_value=0)
            for i in range(0, stepwise_splits.shape[0]):
                xs = set_or_vstack(xs, np.ravel(stepwise_splits[i]), expand_dim_on_set=False)
            ys.extend([[jhmdb_actions.index(action), human_action.id]] * stepwise_splits.shape[0])
        np.savetxt(os.path.join(output_path, "X_{}.csv".format(dataset_part.name.lower())), xs, delimiter=',', fmt='%1.3f')
        np.savetxt(os.path.join(output_path, "y_{}.csv".format(dataset_part.name.lower())), np.asarray(ys, dtype=np.int32),
                   delimiter=',', fmt='%i')

    def __get_feature_vec_producer(self, human_action: HumanAction):
        frame_width = human_action.video_gt.frame_width
        frame_height = human_action.video_gt.frame_height

        if (frame_width, frame_width) not in self.feature_vec_producers:
            self.feature_vec_producers[(frame_width, frame_width)] = FeatureVecProducerSimpleJoints(
                ImageSize(frame_width, frame_height),
                get_joints_func=lambda skeleton: get_joints_jhmdb(skeleton))
        return self.feature_vec_producers[(frame_width, frame_width)]

