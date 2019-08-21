from nobos_commons.data_structures.constants.dataset_part import DatasetPart

from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth
from nobos_dataset_manager.models.view_actiongroundtruth import ViewActionGroundTruth


def test():
    actions_to_load = (ViewActionGroundTruth.select(ViewActionGroundTruth.human_action_id)
                       .where(ViewActionGroundTruth.dataset_name << ['SIM_2019_02_12_idle',
                                                                     'SIM_2019_02_12_walks',
                                                                     'SIM_2019_02_12_walks_Near2cams',
                                                                     'SIM_2019_02_12_waves'],
                              ViewActionGroundTruth.dataset_part == DatasetPart.TRAIN.value,
                              ViewActionGroundTruth.human_datasource == Datasource.OFP_POSE_RECOGNITION.value)
                       )
    a = actions_to_load.count()
    a = 1


def get_video_statistics(datasetname: str):
    dataset = Dataset.get(name=datasetname)
    total_frames = 0
    num_sequences = 0
    for video_gt in dataset.video_ground_truths:
        total_frames += video_gt.num_frames
        num_sequences += 1
        print("{}: FPS: {}, Frames: {}, Width: {}, Height: {}, Created Date: {}".format(video_gt.vid_name, video_gt.fps,
                                                                                        video_gt.num_frames, video_gt.frame_width,
                                                                                        video_gt.frame_height, video_gt.created_date))
    print("Total frames: {}, Total sequences: {}".format(total_frames, num_sequences))
    return total_frames, num_sequences

    # (ViewActionGroundTruth.select(ViewActionGroundTruth.human_action_id)
    #  .distinct()
    #  .where(ViewActionGroundTruth.dataset_name << ['SIM_2019_03_06-idle',
    #                                                'SIM_2019_03_06-jump',
    #                                                'SIM_2019_03_06-sit',
    #                                                'SIM_2019_03_06-walk',
    #                                                'SIM_2019_03_06-wave'],
    #         ViewActionGroundTruth.dataset_part == DatasetPart.TRAIN.value,
    #         ViewActionGroundTruth.human_datasource == datasource.value)
    #  )

if __name__ == "__main__":
    test()
    # test = ["2019_03_13_Freilichtmuseum_Yi_02", "2019_03_13_Freilichtmuseum_Dashcam_02"]
    #
    # ofp_webcam = ['webcam_19_02_05', 'webcam_19_02_06', 'webcam_19_02_12']
    # ofp_record_2019_03_11_30FPS = ['2019_03_11_OFP_Records_Dashcam', '2019_03_11_OFP_Records_Yi']
    # ofp_record_2019_03_11_HSRT_30FPS = ['2019_03_11_OFP_Records_HSRT_Dashcam', '2019_03_11_OFP_Records_HSRT_Yi']
    # ofp_record_2019_03_11_HELLA_30FPS = ['2019_03_11_OFP_Hella_Rec01', '2019_02_27_Hella_OFP_ParkingLot']
    # t2019_03_13_Freilichtmuseum_30FPS = ["2019_03_13_Freilichtmuseum_Yi_01", "2019_03_13_Freilichtmuseum_Dashcam_01"]
    #
    # ofp_from_mocap_30fps = ['SIM_2019_03_06-idle', 'SIM_2019_03_06-walk', 'SIM_2019_03_06-wave']
    # ofp_sim_pose_algo_equal_30fps = ['SIM_2019_02_12_idle', 'SIM_2019_02_12_walks', 'SIM_2019_02_12_walks_Near2cams', 'SIM_2019_02_12_waves']
    # ofp_sim_gt_equal_30fps = ['SIM_2019_02_12_idle', 'SIM_2019_02_12_walks', 'SIM_2019_02_12_walks_Near2cams', 'SIM_2019_02_12_waves']
    # ofp_from_mocap_gt_30fps = ['SIM_2019_03_06-idle', 'SIM_2019_03_06-walk', 'SIM_2019_03_06-wave']
    #
    # all = ofp_webcam + ofp_record_2019_03_11_30FPS + ofp_record_2019_03_11_HSRT_30FPS + ofp_record_2019_03_11_HELLA_30FPS \
    #       + t2019_03_13_Freilichtmuseum_30FPS + ofp_from_mocap_30fps + ofp_from_mocap_gt_30fps + ofp_sim_gt_equal_30fps \
    #         + ofp_sim_pose_algo_equal_30fps
    # real_only = ofp_webcam + ofp_record_2019_03_11_30FPS + ofp_record_2019_03_11_HSRT_30FPS + ofp_record_2019_03_11_HELLA_30FPS \
    #       + t2019_03_13_Freilichtmuseum_30FPS
    # sim_only = ofp_from_mocap_30fps + ofp_from_mocap_gt_30fps + ofp_sim_gt_equal_30fps + ofp_sim_pose_algo_equal_30fps
    #
    # all_frames = 0
    # all_sequences = 0
    # for todo in real_only:
    #     print("---------- {} ------------".format(todo))
    #     total_frames, num_sequences = get_video_statistics(todo)
    #     all_frames += total_frames
    #     all_sequences += num_sequences
    # print("All frames: {}, All sequences: {}".format(all_frames, all_sequences))