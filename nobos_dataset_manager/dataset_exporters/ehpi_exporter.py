import os
from typing import List, Iterator, Dict

import numpy as np
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action, ofp_actions, jhmdb_actions
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_joint_config import \
    get_joints_jhmdb
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_ehpi import \
    FeatureVecProducerEhpi
from nobos_commons.utils.file_helper import get_create_path
from nobos_commons.utils.list_helper import split_list_stepwise

from nobos_dataset_manager.models.view_actiongroundtruth import ViewActionGroundTruth
from nobos_dataset_manager.utils import get_skeleton_from_skeleton_db


class EhpiExporter(object):
    def __init__(self):
        self.skeleton = SkeletonStickman()
        self.feature_vec_producers: Dict[(int, int), FeatureVecProducerEhpi] = {}

    def get_ehpi_images(self, gt_per_action: List[Iterator[ViewActionGroundTruth]], num_actions: int, output_path: str,
                        dataset_part: DatasetPart, actions: List[Action], to_many_zero_frames: int = 5,
                        use_every_n_frame: int = 1,
                        step_size: int = 1):
        output_path = get_create_path(output_path)
        xs: List[List[np.array]] = []
        ys: List[List[int]] = []
        fill_value = np.zeros((15, 3), dtype=np.float32)
        for idx, action_gt_frames in enumerate(gt_per_action):
            feature_vecs: List[np.ndarray] = []
            action = None
            human_action_id = None
            num_frames = None
            print("Working on action sequence ({0}/{1})".format(idx, num_actions))
            # TODO: Use only every 2nd frame for 30fps, take care of the zero split.
            last_processed_frame_num: int = None
            for frame_idx, action_gt in enumerate(action_gt_frames):
                # print("Frame {} in action: {}".format(action_gt.frame_num, action_gt.human_action_id))
                action_gt: ViewActionGroundTruth = action_gt
                if action is None:
                    action = Action(action_gt.action)
                if human_action_id is None:
                    human_action_id = action_gt.human_action_id
                if num_frames is None:
                    num_frames = action_gt.num_frames
                feature_vec_producer = self.__get_feature_vec_producer(action_gt)
                if last_processed_frame_num is None:
                    frame_step = action_gt.frame_num
                else:
                    frame_step = (action_gt.frame_num-1) - last_processed_frame_num
                if frame_step > 0:
                    # IF frames are skipted because theres a no skeleton gt for a frame add zero frames
                    for i in range(0, frame_step):
                        print("SKIPPED!!")
                        feature_vec = np.zeros((feature_vec_producer.num_joints, 3), dtype=np.float32)
                        feature_vecs.append(feature_vec)
                skeleton = get_skeleton_from_skeleton_db(action_gt)
                feature_vec = feature_vec_producer.get_feature_vec(skeleton)
                feature_vecs.append(feature_vec)
                last_processed_frame_num = action_gt.frame_num
            #     # TODO: SPLIT TAKES FOOOOCKING LONG
            # a = int(num_frames / use_every_n_frame)
            for i in range(int(num_frames / use_every_n_frame) - len(feature_vecs)):
                feature_vec = np.zeros((feature_vec_producer.num_joints, 3), dtype=np.float32)
                feature_vecs.append(feature_vec)
            if int(num_frames / use_every_n_frame) != num_frames:
                raise Exception("Something is very wrong")
            print("Split sets")
            stepwise_splits = split_list_stepwise(feature_vecs, split_size=32, step_size=step_size,
                                                  fill_value=fill_value,
                                                  every_n_element=use_every_n_frame)
            print("Remove zeros")
            if to_many_zero_frames is not None:
                split_idxs_with_to_many_zeros = []
                for split_idx, split in enumerate(stepwise_splits):
                    count_zeros = 0
                    for split_array in split:
                        if split_array.max() == 0:
                            count_zeros += 1
                        if count_zeros > to_many_zero_frames:  # More than 5 frames missing
                            split_idxs_with_to_many_zeros.append(split_idx)
                            break
                # Remove elements by idx in reverse order that no idx changes on later elements appear
                for split_to_delete_idx in sorted(split_idxs_with_to_many_zeros, reverse=True):
                    del stepwise_splits[split_to_delete_idx]
            xs.extend(stepwise_splits)
            ys.extend([[actions.index(action), human_action_id]] * len(stepwise_splits))
        xs_array = np.array(xs, dtype=np.float32)
        xs_array = np.reshape(xs_array, (xs_array.shape[0], xs_array.shape[1] * xs_array.shape[2] * xs_array.shape[3]))
        ys_array = np.array(ys, dtype=np.int32)
        np.savetxt(os.path.join(output_path, "X_{}.csv".format(dataset_part.name.lower())), xs_array, delimiter=',',
                   fmt='%1.3f')
        np.savetxt(os.path.join(output_path, "y_{}.csv".format(dataset_part.name.lower())),
                   np.asarray(ys_array, dtype=np.int32),
                   delimiter=',', fmt='%i')

    def __get_feature_vec_producer(self, action_gt: ViewActionGroundTruth):
        frame_width = action_gt.frame_width
        frame_height = action_gt.frame_height
        if (frame_width, frame_width) not in self.feature_vec_producers:
            # TODO: Normalize here? Is bad because hard to trasform on train.. make sure that image size is available at train time
            self.feature_vec_producers[(frame_width, frame_width)] = FeatureVecProducerEhpi(
                ImageSize(frame_width, frame_height),
                get_joints_func=lambda skeleton: get_joints_jhmdb(skeleton))
        return self.feature_vec_producers[(frame_width, frame_width)]
