import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from nobos_commons.utils.file_helper import get_create_path


def get_label_statistics(y: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(y[:, 0], return_counts=True)
    return dict(zip(unique, counts))


def create_equal_distributed_dataset(X_path_in: str, y_path_in: str, X_path_out: str, y_path_out: str):
    print("Load X")
    x = pd.read_csv(X_path_in, delimiter=",", dtype=np.float32).values
    # x = np.loadtxt(X_path_in, delimiter=',', dtype=np.float32)
    print("Load y")
    y = np.loadtxt(y_path_in, delimiter=',', dtype=np.int32)
    # test = np.where(y[:, 0] == 0)
    # test2 = np.where(y[:, 0] == 1)
    # test3 = np.where(y[:, 0] == 2)

    label_statistics = get_label_statistics(y)
    split_size = label_statistics[min(label_statistics, key=label_statistics.get)]

    # # stuff_to_delete = np.where(x[:].max() == 0)
    # deletetest = np.where(~x.any(axis=1))[0]

    indexes_1 = random.sample(np.where(y[:] == 0)[0].tolist(), split_size)
    indexes_2 = random.sample(np.where(y[:] == 1)[0].tolist(), split_size)
    indexes_3 = random.sample(np.where(y[:] == 2)[0].tolist(), split_size)
    x_1 = np.take(x, np.array(indexes_1), axis=0)
    x_2 = np.take(x, np.array(indexes_2), axis=0)
    x_3 = np.take(x, np.array(indexes_3), axis=0)
    x = np.vstack((x_1, x_2, x_3))
    y_1 = np.take(y, np.array(indexes_1), axis=0)
    y_2 = np.take(y, np.array(indexes_2), axis=0)
    y_3 = np.take(y, np.array(indexes_3), axis=0)
    y = np.vstack((y_1, y_2, y_3))
    np.savetxt(X_path_out, x, delimiter=',', fmt='%1.3f')
    np.savetxt(y_path_out, y, delimiter=',', fmt='%i')

if __name__ == "__main__":
    X_in = "/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_pose_algo_30fps/X_train.csv"
    y_in = "/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_pose_algo_30fps/y_train.csv"
    X_out = os.path.join(get_create_path("/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_equal_pose_algo_30fps"), "X_train.csv")
    y_out = os.path.join(get_create_path("/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_equal_pose_algo_30fps"), "y_train.csv")
    create_equal_distributed_dataset(X_in, y_in, X_out, y_out)

    X_in = "/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_gt_30fps/X_train.csv"
    y_in = "/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_gt_30fps/y_train.csv"
    X_out = os.path.join(get_create_path("/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_equal_gt_30fps"), "X_train.csv")
    y_out = os.path.join(get_create_path("/media/disks/beta/dataset_src/ehpi/ofp_from_mocap_equal_gt_30fps"), "y_train.csv")
    create_equal_distributed_dataset(X_in, y_in, X_out, y_out)