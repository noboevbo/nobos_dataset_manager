import os
from _ast import Dict

import numpy as np
from nobos_commons.utils.file_helper import get_immediate_subdirectories, get_last_dir_name

action_mapping = {
    "idle": 0,
    "rom": 0,
    "walk": 0,
    "wave": 1
}


def vecs_per_frame_to_time_frame_vecs(feature_vecs_per_frame: list, time_frame_length) -> list:
    """
    Takes the feature vectors for n frames and creates new vectors for time_frames of length time_frame_length by
    appending the features. Used for classic ML
    :param feature_vecs_per_frame:
    :param time_frame_length:
    :return:
    """
    full_list = []
    for i in range(0, len(feature_vecs_per_frame) - time_frame_length):
        time_window_vecs = None
        count_zeros = 0
        for k, j in enumerate(range(i, i + time_frame_length)):
            if feature_vecs_per_frame[j].max() == 0:
                count_zeros += 1
            if time_window_vecs is None:
                time_window_vecs = feature_vecs_per_frame[j]
            else:
                time_window_vecs = np.vstack((time_window_vecs, feature_vecs_per_frame[j]))
        if time_window_vecs.max() == 0 or count_zeros > 25:
            continue
        full_list.append(time_window_vecs)
    full_array = np.array(full_list)
    return full_array

if __name__ == "__main__":
    src_dir = "/media/disks/beta/datasets/2019_24_01_action"
    arrays = None
    ys = []
    for sub_dir in get_immediate_subdirectories(src_dir):
        action = get_last_dir_name(sub_dir)
        array_action = None
        for file_path in os.listdir(sub_dir):
            keypoints = np.loadtxt(os.path.join(sub_dir, file_path), delimiter=',')
            if len(keypoints) == 0:
                a = 1
            # blocks = int(len(keypoints) / 32)
            # X_ = np.array(np.split(keypoints[:blocks*32,:], blocks))
            # X_ = keypoints
            # X_[:, :, ::2] *= (1 / np.max(X_[:, :, ::2]))
            # X_[:, :, 1::2] *= (1 / np.max(X_[:, :, 1::2]))
            # if arrays is None:
            #     arrays = X_
            # else:
            #     arrays = np.vstack((arrays, X_))
            X_ = vecs_per_frame_to_time_frame_vecs(keypoints, 32)
            if X_ is None or len(X_.shape) < 2:
                continue
            # try:
            #     if X_.shape[2] != 38:
            #         a = 1
            # except:
            #     a = 1
            # try:
            #     X_[:, :, ::2] *= (1 / np.max(X_[:, :, ::2]))
            #     X_[:, :, 1::2] *= (1 / np.max(X_[:, :, 1::2]))
            # except:
            #     a = 1
            if array_action is None:
                array_action = X_
            else:
                array_action = np.vstack((array_action, X_))
            # if ys is None:
            #     ys = np.ones((X_.shape[0])) * action_mapping[action]
            # else:
            #     ys.extend(action_mapping[action]
            #     try:
            #         ys = np.vstack((ys, results))
            #     except:
            #         a = 1
        if action == "idle":
            array_action = array_action[np.random.randint(array_action.shape[0], size=12000),:]
        ys.extend([action_mapping[action]+1] * array_action.shape[0])
        if arrays is None:
            arrays = array_action
        else:
            arrays = np.vstack((arrays, array_action))

    arrays[arrays < 0] = 0 # Remove negative values
    print(ys.count(1))
    print(ys.count(2))
    print(ys.count(3))
    reshaped = arrays.reshape((arrays.shape[0]*32, 38))
    np.savetxt(os.path.join("/media/disks/beta/datasets/2019_24_01_action", "X_train.txt"), reshaped, delimiter=',',
               fmt='%1.3f')
    np.savetxt(os.path.join("/media/disks/beta/datasets/2019_24_01_action", "Y_train.txt"), np.asarray(ys, dtype=np.int32), delimiter=',',
               fmt='%i')
    a =1

