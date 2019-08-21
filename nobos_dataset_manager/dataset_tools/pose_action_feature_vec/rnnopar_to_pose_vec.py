import os

import numpy as np
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_ehpi import \
    FeatureVecProducerEhpi
from nobos_commons.utils.human_surveyor import HumanSurveyor


class RnnOpArConverter(object):
    def __init__(self, src_dataset_path: str, target_dataset_path: str, feature_vec_producer):
        """
        Converts RnnOpAr Skeleton (OP) to Stickman and saves it in same structure

        Note: They just split the Train / Test Data by 32 steps, so they have overlaps of subjects, actions etc. in the
        data. This imports it as it is to make it comparable, but at least one should convert the data which has interclass overlaps
        :param src_dataset_path: The path to the JHMDB root folder
        """
        self.dataset_path = src_dataset_path
        self.target_dataset_path = target_dataset_path
        self.feature_vec_producer = feature_vec_producer
        self.X_train_path = os.path.join(self.dataset_path, "X_train.txt")
        self.X_test_path = os.path.join(self.dataset_path, "X_test.txt")

        self.y_train_path = os.path.join(self.dataset_path, "Y_train.txt")
        self.y_test_path = os.path.join(self.dataset_path, "Y_test.txt")
        #
        # # TOODO: THIS IN COMMONS OR SO
        # self.action_mapping: Dict[int, str] = {
        #     1: "jumping",
        #     2: "jumping_jacks",
        #     3: "boxing",
        #     4: "waving_two_hands",
        #     5: "waving_one_hand",
        #     6: "clapping_hands"
        # }

    def import_data(self):
        # Load the networks inputs
        # X_train = self.__convert_X(self.X_train_path, "X_train.txt")
        X_test = self.__convert_X(self.X_test_path, "X_test.txt")

    def __convert_X(self, X_path, file_name):
        skeleton_arrays = []
        with open(X_path, 'r') as file:
            file_list = list(file)
            length = len(file_list)
            for row_index, row in enumerate(file_list):
                print("Loading X frame {0}/{1}".format(row_index, length))
                skeleton = SkeletonStickman()
                splits = row.split(',')
                for joint_index, column_index in enumerate(range(0, len(splits), 2)):
                    skeleton.joints[joint_index].x = float(splits[column_index])
                    skeleton.joints[joint_index].y = float(splits[column_index + 1])
                    skeleton.joints[joint_index].score = 1
                    if skeleton.joints[joint_index].x <= 0 or skeleton.joints[joint_index].y <= 0:
                        skeleton.joints[joint_index].score = 0
                skeleton_feature_vec = self.feature_vec_producer.get_feature_vec(skeleton)
                # skeleton = skeleton.joints.to_numpy()
                skeleton_arrays.append(skeleton_feature_vec)
        np.savetxt(os.path.join(self.target_dataset_path, file_name), np.asarray(skeleton_arrays), delimiter=',',
                   fmt='%1.3f')

if __name__ == "__main__":
    # loader = RnnOpArConverter("/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/keypoints",
    #                           "/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/feature_vecs",
    #                           FeatureVectorProducerFangEtAl(SkeletonStickman(), HumanSurveyor()))
    loader = RnnOpArConverter("/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/keypoints",
                              "/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/feature_vecs_ehpi",
                              FeatureVecProducerEhpi(SkeletonStickman(), ImageSize(1280, 720), HumanSurveyor()))
    loader.import_data()
