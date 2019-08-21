import os
from typing import Dict

import numpy as np
from nobos_commons.data_structures.skeletons.skeleton_openpose import SkeletonOpenPose
from nobos_commons.tools.skeleton_converters.skeleton_converter_openpose_to_stickman import \
    SkeletonConverterOpenPoseToStickman
from nobos_commons.utils.human_surveyor import HumanSurveyor


class RnnOpArConverter(object):
    def __init__(self, src_dataset_path: str, target_dataset_path: str, skeleton_converter: SkeletonConverterOpenPoseToStickman):
        """
        Converts RnnOpAr Skeleton (OP) to Stickman and saves it in same structure

        Note: They just split the Train / Test Data by 32 steps, so they have overlaps of subjects, actions etc. in the
        data. This imports it as it is to make it comparable, but at least one should convert the data which has interclass overlaps
        :param src_dataset_path: The path to the JHMDB root folder
        """
        self.dataset_path = src_dataset_path
        self.target_dataset_path = target_dataset_path
        self.skeleton_converter = skeleton_converter
        self.X_train_path = os.path.join(self.dataset_path, "X_train.txt")
        self.X_test_path = os.path.join(self.dataset_path, "X_test.txt")

        self.y_train_path = os.path.join(self.dataset_path, "Y_train.txt")
        self.y_test_path = os.path.join(self.dataset_path, "Y_test.txt")

        self.human_surveyor = HumanSurveyor()

        # TOODO: THIS IN COMMONS OR SO
        self.action_mapping: Dict[int, str] = {
            1: "jumping",
            2: "jumping_jacks",
            3: "boxing",
            4: "waving_two_hands",
            5: "waving_one_hand",
            6: "clapping_hands"
        }

    def import_data(self):
        # Load the networks inputs
        X_train = self.__convert_X(self.X_train_path, "train.txt")
        X_test = self.__convert_X(self.X_test_path, "test.txt")

    def __convert_X(self, X_path, file_name):
        skeleton_arrays = []
        with open(X_path, 'r') as file:
            file_list = list(file)
            length = len(file_list)
            for row_index, row in enumerate(file_list):
                print("Loading X frame {0}/{1}".format(row_index, length))
                skeleton = SkeletonOpenPose()
                splits = row.split(',')
                for joint_index, column_index in enumerate(range(0, len(splits), 2)):
                    skeleton.joints[joint_index].x = float(splits[column_index])
                    skeleton.joints[joint_index].y = float(splits[column_index + 1])
                    skeleton.joints[joint_index].score = 1
                    if skeleton.joints[joint_index].x <= 0 or skeleton.joints[joint_index].y <= 0:
                        skeleton.joints[joint_index].score = 0
                skeleton = self.skeleton_converter.get_converted_skeleton(skeleton)
                skeleton = skeleton.joints.to_numpy()
                skeleton_arrays.append(skeleton)
        np.savetxt(os.path.join(self.target_dataset_path, file_name), np.asarray(skeleton_arrays), delimiter=',',
                   fmt='%1.3f')

if __name__ == "__main__":
    loader = RnnOpArConverter("/home/dennis/Downloads/RNN-HAR-2D-Pose-database", "/media/disks/beta/datasets/rnnopar",
                              SkeletonConverterOpenPoseToStickman())
    loader.import_data()
