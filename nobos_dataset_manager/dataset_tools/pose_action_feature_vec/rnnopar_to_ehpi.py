import os

import numpy as np
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.utils.human_surveyor import HumanSurveyor

from nobos_dataset_manager.dataset_tools.pose_action_feature_vec.feature_vec_producer_ehpi import \
    FeatureVectorProducerEhpi


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
        X_train = self.__convert_X(self.X_train_path, self.y_train_path, "X_train.txt")
        # X_test = self.__convert_X(self.X_test_path, self.y_test_path, "X_test.txt")

    def __convert_X(self, X_path, Y_path, file_name):
        skeleton_arrays = []
        y_out = []
        with open(X_path, 'r') as file:
            with open(Y_path, 'r') as yfile:
                ys = list(yfile)
                file_list = list(file)
                length = len(file_list)
                for gt_idx, i in enumerate(range(0, len(file_list), 32)):
                    batch_arrays = []
                    print("Batch {}, GT: {}".format(i, gt_idx))
                    for vec_idx in range(i, i+32):
                        row = file_list[vec_idx]
                        print("Loading X frame {0}/{1}".format(vec_idx, length))
                        skeleton = SkeletonStickman()
                        splits = row.split(',')
                        for joint_index, column_index in enumerate(range(0, len(splits), 3)):
                            skeleton.joints[joint_index].x = float(splits[column_index])
                            if skeleton.joints[joint_index].x < 0:
                                skeleton.joints[joint_index].x = 0
                            skeleton.joints[joint_index].y = float(splits[column_index + 1])
                            if skeleton.joints[joint_index].y < 0:
                                skeleton.joints[joint_index].y = 0
                            skeleton.joints[joint_index].score = float(splits[column_index + 2])
                            # if skeleton.joints[joint_index].x <= 0 or skeleton.joints[joint_index].y <= 0:
                            #     skeleton.joints[joint_index].score = 0
                        skeleton_feature_vec = self.feature_vec_producer.get_feature_vec(skeleton)
                        # skeleton = skeleton.joints.to_numpy()
                        skeleton_feature_vec = np.ravel(skeleton_feature_vec)
                        batch_arrays.append(skeleton_feature_vec)
                    test1 = np.array(batch_arrays)
                    if test1[:, 1::3].max() == 0 or test1[:, 0::3].max() == 0:
                        continue
                    skeleton_arrays.extend(batch_arrays)
                    y_out.append(ys[gt_idx])
                # for row_index, row in enumerate(file_list):
                #     print("Loading X frame {0}/{1}".format(row_index, length))
                #     skeleton = SkeletonStickman()
                #     splits = row.split(',')
                #     for joint_index, column_index in enumerate(range(0, len(splits), 3)):
                #         skeleton.joints[joint_index].x = float(splits[column_index])
                #         skeleton.joints[joint_index].y = float(splits[column_index + 1])
                #         skeleton.joints[joint_index].score = float(splits[column_index + 2])
                #         # if skeleton.joints[joint_index].x <= 0 or skeleton.joints[joint_index].y <= 0:
                #         #     skeleton.joints[joint_index].score = 0
                #     skeleton_feature_vec = self.feature_vec_producer.get_feature_vec(skeleton)
                #     if skeleton_feature_vec.max() == 0:
                #         print("jo")
                #         continue
                #     # skeleton = skeleton.joints.to_numpy()
                #     skeleton_feature_vec = np.ravel(skeleton_feature_vec)
                #     skeleton_arrays.append(skeleton_feature_vec)
                #     y_out.append(ys[row_index])
        np.savetxt(os.path.join(self.target_dataset_path, file_name), np.asarray(skeleton_arrays), delimiter=',',
                   fmt='%1.3f')
        np.savetxt(os.path.join(self.target_dataset_path, file_name+"ys"), np.asarray(y_out, dtype=np.int32), delimiter=',',
                   fmt='%i')

if __name__ == "__main__":
    # loader = RnnOpArConverter("/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/keypoints",
    #                           "/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/feature_vecs",
    #                           FeatureVectorProducerFangEtAl(SkeletonStickman(), HumanSurveyor()))
    loader = RnnOpArConverter("/home/dennis/sync/cogsys/datasets/ehpi/jhmdb/32frames/keypoints",
                              "/home/dennis/sync/cogsys/datasets/ehpi/jhmdb/32frames",
                              FeatureVectorProducerEhpi(SkeletonStickman(), ImageSize(1280, 720), HumanSurveyor()))
    loader.import_data()
