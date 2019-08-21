from typing import Dict

import numpy as np
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_fangetal import \
    FeatureVectorFangEtAl


class FeatureVectorSequencer(object):
    __slots__ = ['feature_vec_length']

    def __init__(self, feature_vec_length: int):
        self.feature_vec_length = feature_vec_length

    @staticmethod
    def __get_dict_frame_id_index(feature_vec_per_frame: Dict[int, FeatureVectorFangEtAl]):
        return [key for key in sorted(feature_vec_per_frame.keys())]

    def vecs_per_frame_to_time_frame_vecs(feature_vecs_per_frame: list, time_frame_length) -> np.ndarray:
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
                if feature_vecs_per_frame[j].max() < 2:
                    count_zeros += 1
                if time_window_vecs is None:
                    time_window_vecs = feature_vecs_per_frame[j]
                else:
                    time_window_vecs = np.vstack((time_window_vecs, feature_vecs_per_frame[j]))
            if time_window_vecs.max() == 0 or count_zeros > 5:
                continue
            full_list.append(time_window_vecs)
        full_array = np.array(full_list)
        return full_array

    def get_feature_vecs_for_timeframe(self, feature_vec_per_frame: Dict[int, FeatureVectorFangEtAl],
                                                    timeframe_length: int) -> list:
        """
        Takes the feature vectors for n frames and creates new vectors for time_frames of length time_frame_length by
        appending the features. Used for classic ML.
        e.g. FeatureVec with Length 1456 and time_frame_length of 5 will result in a concatenated vec of length 5 * 1456
        :param feature_vec_per_frame: the feature vectors per frame
        :param timeframe_length: timeframe length in frames
        :return:
        """
        frame_id_index = self.__get_dict_frame_id_index(feature_vec_per_frame)
        time_frame_feature_vecs = []
        for frame_num in range(timeframe_length, len(feature_vec_per_frame)):
            time_frame_feature_vec = np.empty([timeframe_length * self.feature_vec_length])
            for timeframe_vec_pos, frame_id_num in enumerate(range(frame_num - timeframe_length, frame_num)):
                feature_vec_frame_nr = frame_id_index[frame_id_num]
                feature_vec = feature_vec_per_frame[feature_vec_frame_nr]
                from_vec_pos = timeframe_vec_pos * self.feature_vec_length
                to_vec_pos = timeframe_vec_pos * self.feature_vec_length + self.feature_vec_length
                time_frame_feature_vec[from_vec_pos:to_vec_pos] = feature_vec.feature_vec
            time_frame_feature_vecs.append(time_frame_feature_vec)
        return time_frame_feature_vecs

    def get_rnn_sequences(self, feature_vec_per_frame: Dict[int, FeatureVectorFangEtAl], timeframe_length):
        """
        Returns the feature vecs in rnn inpute sequence format for the given timeframe_length
        :param feature_vec_per_frame: Feature vector for each frame -> 450 frames  -> len(feature_vecs) == 450
        :param time_frame_length: length of the time frame input for the RNN
        :return: numpy_array with shape [time_frame_length, 1, self.feature_vec_length]
        """
        frame_id_index = self.__get_dict_frame_id_index(feature_vec_per_frame)
        rnn_input_sequences = []
        for frame_num in range(timeframe_length, len(feature_vec_per_frame)):
            rnn_input_sequence = np.empty([timeframe_length, 1, self.feature_vec_length])
            # remove the batch_size because in the last view frames there would be empty slots
            for timeframe_vec_pos, frame_id_num in enumerate(range(frame_num - timeframe_length, frame_num)):
                feature_vec_frame_nr = frame_id_index[frame_id_num]
                feature_vec = feature_vec_per_frame[feature_vec_frame_nr]
                if feature_vec is None:
                    # TODO: If no humans where in img the feature vec is none and only a 0 vec is used, better handling?
                    rnn_input_sequence[timeframe_vec_pos][0] = np.zeros([self.feature_vec_length])
                else:
                    rnn_input_sequence[timeframe_vec_pos][0] = feature_vec.feature_vec
            rnn_input_sequences.append(rnn_input_sequence)
        return rnn_input_sequences
