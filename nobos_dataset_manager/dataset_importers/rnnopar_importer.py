# import os
# import pymongo
# from typing import List, Dict
#
# from nobos_commons.data_structures.skeletons.skeleton_openpose import SkeletonOpenPose
# from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
# from nobos_commons.tools.log_handler import logger
# from nobos_commons.tools.skeleton_converters.skeleton_converter_openpose_to_stickman import \
#     SkeletonConverterOpenPoseToStickman
# from nobos_commons.utils.list_helper import get_chunks_by_list_sampler, split_list
# from pymongo import MongoClient
#
#
# class RnnOpArImporter(object):
#     def __init__(self, dataset_path: str, skeleton_converter: SkeletonConverterOpenPoseToStickman):
#         """
#         Importer for pose based action recognition data from https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
#         Dataset Structure:
#         X files: Pose Data in MS OpenPose Format j0_x, j0_y, j1_x, j1_y , j2_x, [...]
#
#         Note: They just split the Train / Test Data by 32 steps, so they have overlaps of subjects, actions etc. in the
#         data. This imports it as it is to make it comparable, but at least one should convert the data which has interclass overlaps
#         :param dataset_path: The path to the JHMDB root folder
#         """
#         self.dataset_path = dataset_path
#         self.skeleton_converter = skeleton_converter
#         self.n_steps = 32
#         self.X_train_path = os.path.join(self.dataset_path, "X_train.txt")
#         self.X_test_path = os.path.join(self.dataset_path, "X_test.txt")
#
#         self.y_train_path = os.path.join(self.dataset_path, "Y_train.txt")
#         self.y_test_path = os.path.join(self.dataset_path, "Y_test.txt")
#
#         # TOODO: THIS IN COMMONS OR SO
#         self.action_mapping: Dict[int, str] = {
#             1: "jumping",
#             2: "jumping_jacks",
#             3: "boxing",
#             4: "waving_two_hands",
#             5: "waving_one_hand",
#             6: "clapping_hands"
#         }
#
#     def import_data(self):
#         # Load the networks inputs
#         X_train = self.__load_X(self.X_train_path)
#         X_test = self.__load_X(self.X_test_path)
#
#         # print X_test
#
#         y_train = self.__load_y(self.y_train_path)
#         y_test = self.__load_y(self.y_test_path)
#
#         gt_list = []
#         for split_nr, train_data in enumerate(X_train):
#             creation_data = os.path.getmtime(self.X_train_path)
#             train_gt = RnnOpArGroundTruth(unique_id="{0}_{1}".format(DatasetPartType.TRAIN.name, str(split_nr)),
#                                           creation_date=creation_data,
#                                           action=y_train[split_nr],
#                                           frames=train_data,
#                                           dataset_part=DatasetPart("RnnOpAr", DatasetPartType.TRAIN))
#             gt_list.append(train_gt)
#
#         for split_nr, test_data in enumerate(X_test):
#             creation_data = os.path.getmtime(self.X_test_path)
#             test_gt = RnnOpArGroundTruth(unique_id="{0}_{1}".format(DatasetPartType.TEST.name, str(split_nr)),
#                                          creation_date=creation_data,
#                                          action=y_test[split_nr],
#                                          frames=test_data,
#                                          dataset_part=DatasetPart("RnnOpAr", DatasetPartType.TEST))
#             gt_list.append(test_gt)
#
#         logger.info('Write to database...')
#         db_client = MongoClient(username="ofp_user", password="ofp2019dem0!")
#         db = db_client.ground_truth_store
#         rnnopar_db = db.rnnopar
#         rnnopar_db.create_index([('uid', pymongo.ASCENDING)], unique=True)
#
#         for num, gt in enumerate(gt_list):
#             logger.info('{0}/{1}'.format(num, len(gt_list)))
#             rnnopar_db.update({'uid': gt.uid}, gt.to_dict(), upsert=True)
#
#     def __load_X(self, X_path):
#         skeletons: List[SkeletonStickman] = []
#         with open(X_path, 'r') as file:
#             file_list = list(file)
#             length = len(file_list)
#             for row_index, row in enumerate(file_list):
#                 print("Loading X frame {0}/{1}".format(row_index, length))
#                 skeleton = SkeletonOpenPose()
#                 splits = row.split(',')
#                 for joint_index, column_index in enumerate(range(0, len(splits), 2)):
#                     skeleton.joints[joint_index].x = float(splits[column_index])
#                     skeleton.joints[joint_index].y = float(splits[column_index + 1])
#                     skeleton.joints[joint_index].score = 1
#                     if skeleton.joints[joint_index].x <= 0 or skeleton.joints[joint_index].y <= 0:
#                         skeleton.joints[joint_index].score = 0
#                 skeletons.append(self.skeleton_converter.get_converted_skeleton(skeleton))
#         skeleton_splits = split_list(skeletons, 32)
#         return skeleton_splits
#
#     def __load_y(self, y_path):
#         actions: List[str] = []
#         with open(y_path, 'r') as file:
#             file_list = list(file)
#             length = len(file_list)
#             for row_index, row in enumerate(file_list):
#                 print("Loading y frame {0}/{1}".format(row_index, length))
#                 splits = row.replace('  ', ' ').strip().split(' ')
#                 for action in splits:
#                     actions.append(self.action_mapping[int(action)])
#         return actions
#
#
# if __name__ == "__main__":
#     loader = RnnOpArImporter("/home/dennis/Downloads/RNN-HAR-2D-Pose-database", SkeletonConverterOpenPoseToStickman())
#     loader.import_data()
