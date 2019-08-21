from typing import List

import numpy as np

from nobos_dataset_manager.dataset_tools.pose_action_feature_vec.data.three_joint_features import ThreeJointFeatures
from nobos_dataset_manager.dataset_tools.pose_action_feature_vec.data.two_joint_features import TwoJointFeatures


class FeatureVectorFangEtAl(object):
    two_joint_features_list: List[TwoJointFeatures]
    three_joint_features_list: List[ThreeJointFeatures]

    @property
    def feature_vec(self) -> np.ndarray:
        feature_vec: List[float] = []
        for two_joint_features in self.two_joint_features_list:
            feature_vec.append(two_joint_features.normalized_distance_x)
            feature_vec.append(two_joint_features.normalized_distance_y)
            feature_vec.append(two_joint_features.normalized_distance_euclidean)
            feature_vec.append(two_joint_features.angle_rad)
        for three_joint_features in self.three_joint_features_list:
            feature_vec.append(three_joint_features.angle_alpha_rad)
            feature_vec.append(three_joint_features.angle_beta_rad)
            feature_vec.append(three_joint_features.angle_gamma_rad)
        return np.asarray(feature_vec)

    def __init__(self, two_joint_features_list: List[TwoJointFeatures] = None,
                 three_joint_features_list: List[ThreeJointFeatures] = None):
        self.two_joint_features_list = two_joint_features_list if two_joint_features_list is not None else []
        self.three_joint_features_list = three_joint_features_list if three_joint_features_list is not None else []
