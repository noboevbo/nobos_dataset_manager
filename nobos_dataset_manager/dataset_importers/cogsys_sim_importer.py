import os
import pickle
from datetime import datetime
from typing import Dict
from nobos_dataset_manager import configurator
configurator.setup()

import pandas as pd
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.input_providers.camera.img_dir_provider import ImgDirProvider
from nobos_commons.tools.file_manager import transfer_files
from nobos_commons.tools.log_handler import logger
from nobos_commons.utils.action_helper import get_action_from_string
from nobos_commons.utils.file_helper import get_immediate_subdirectories, get_last_dir_name, \
    get_create_path, batch_rename_files_to_index
from nobos_commons.utils.human_helper import get_human_with_highest_score
from nobos_dataset_api.rtsim.rtsim_pose import get_skeleton_3D_from_row, read_annotation_3d


from nobos_dataset_manager.config import cfg
from nobos_dataset_manager.dataset_tools.pipeline_ehpi import PipelineEhpi
from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.dataset_split import DatasetSplit
from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.frame_ground_truth import FrameGroundTruth
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth
from nobos_dataset_manager.utils import get_skeleton_db_from_skeleton, get_bb_db_from_skeleton, \
    get_bb_3D_db_from_skeleton


# TODO: Test

class TempAction(object):
    def __init__(self, action_name: str, action_db: HumanAction):
        self.action_name: str = action_name
        self.action_db: HumanAction = action_db

class CogsysSimImporter(object):
    def __init__(self, sim_record_path: str, dataset_name: str, image_size: ImageSize, fps: int,
                 pose_tracking: bool, direct_action_mapping: Dict[str, Action], dataset_part: DatasetPart):
        """
        This class currently supports only 1 human in the image..
        So we have Path -> Foldername[==ActionName] -> n .pngs and 1 joint_positions.csv

        Importer for a Pose Algorithm results. The results should be written in a csv with all joint positions and
        scores in a row and one row per human per frame. 
        frame_num, human_id, action, x_1_1, y_1_1, score_1_1, [...] x_1_n, y_1_n, score_1_n
        frame_num, human_id, action, x_2_1, y_2_1, score_2_1, [...] x_2_n, y_2_n, score_2_n
        :param dataset_name: THe name of the dataset
        :param pose_result_path: The path to the joint_positions.csv
        :param src_image_dir: The path to the images on which the pose results are produced on
        :param vid_path: Path to the video of the src image dir
        """
        # TODO: First import img dir to video_gt, then run pose rec algo? To make sure we have the same indexes etc.
        assert os.path.exists(sim_record_path), "simulation record folder '{}' could not be found!".format(
            sim_record_path)

        self.sim_record_path = sim_record_path
        self.img_dir_to_skeleton_tool = PipelineEhpi(use_pose_tracking=pose_tracking, use_fast_mode=True, image_size=image_size)
        self.fps = fps
        self.image_size = image_size
        self.dataset, created = Dataset.get_or_create(name=dataset_name)
        self.dataset_split, create = DatasetSplit.get_or_create(name="{}-1".format(dataset_name), dataset_part=dataset_part.value)
        self.direct_action_mapping = direct_action_mapping
        if created:
            print("Created new dataset DB entry: '{0}'".format(self.dataset.name))

        self.img_blob_path = os.path.join('data', 'video_gt', dataset_name, 'imgs')
        self.segmentation_blob_path = os.path.join('data', 'video_gt', dataset_name, 'segs')
        self.depth_blob_path = os.path.join('data', 'video_gt', dataset_name, 'depths')
        self.vid_blob_path = os.path.join('data', 'video_gt', dataset_name, 'vids')
        self.imgs_path = get_create_path(os.path.join(cfg.blob_storage_path, self.img_blob_path))
        self.segmentations_path = get_create_path(os.path.join(cfg.blob_storage_path, self.segmentation_blob_path))
        self.depths_path = get_create_path(os.path.join(cfg.blob_storage_path, self.depth_blob_path))
        self.vids_path = get_create_path(os.path.join(cfg.blob_storage_path, self.vid_blob_path))

        self.datasource_gt = Datasource.GROUND_TRUTH
        self.datasource_algo = Datasource.OFP_POSE_RECOGNITION

    def __save_frame_contents_from_gt(self, gt_df: pd.DataFrame, video_gt: VideoGroundTruth):
        logger.info("Import data from ground truth...")
        frame_gts: Dict[int, FrameGroundTruth] = {}
        human_actions: Dict[str, TempAction] = {}
        for index, row in gt_df.iterrows():
            frame_num = row["frame_num"]
            if frame_num not in frame_gts:
                # Save Frame
                frame_gt, created = FrameGroundTruth.get_or_create(frame_num=frame_num, video_gt=video_gt)
                if created:
                    print("Created new frame_gt")
                frame_gts[frame_num] = frame_gt

            human_uid = row['uid']
            action_name = row['action']
            if action_name in self.direct_action_mapping:
                action = self.direct_action_mapping[action_name]
            else:
                action = get_action_from_string(action_name)
            # Action
            if human_uid not in human_actions or action_name != human_actions[human_uid].action_name:
                action_db = HumanAction()
                action_db.action = action.value
                action_db.video_gt = video_gt
                action_db.save()
                human_actions[human_uid] = TempAction(action_name, action_db)

            # Save Human
            human_db = Human()
            human_db.uid = human_uid
            human_db.frame_gt = frame_gts[frame_num]
            human_db.action = human_actions[human_uid].action_db
            human_db.datasource = self.datasource_gt.value
            human_db.save()

            skeleton = get_skeleton_3D_from_row(row)
            # Save Skeleton
            skeleton_joints = get_skeleton_db_from_skeleton(skeleton, human_db)
            skeleton_joints.save()

            # Save BB
            bb = get_bb_3D_db_from_skeleton(skeleton, human_db)
            bb.save()

    def __save_frame_contents_from_pose_rec(self, cam_dir_path: str, gt_df: pd.DataFrame, video_gt: VideoGroundTruth):
        logger.info("Import data from pose recognition...")
        cache_file_path = os.path.join(cam_dir_path, "image_content.pkl")
        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as file:
                frame_contents = pickle.load(file)
        else:
            img_dir_provider = ImgDirProvider(cam_dir_path, image_size=self.image_size, print_current_path=True)
            frame_contents = self.img_dir_to_skeleton_tool.get_from_input_provider(img_dir_provider)
            pickle.dump(frame_contents, open(cache_file_path, 'wb'), protocol=4)
        # TODO: This frame contents is from algo, we can also import from GT!
        # TODO: Read Pose GT for the action sequences, add for each human the human.action from GT!
        # TODO: This is not correct, because in gt one row is only per human per frame, not per frame..
        num_gt_frames = gt_df['frame_num'].max() + 1
        if len(gt_df) != num_gt_frames:
            raise NotImplementedError("Multiple humans per frame in GT, this is not supported, because it requires to"
                                      "detect which human in gt belongs to which human in pose algo output. Cam Path: "
                                      "'{0}'".format(cam_dir_path))
        if len(frame_contents) != num_gt_frames:
            raise KeyError("Length of frame contents '{0}' ({1}) differ from length of ground truth {2}"
                           .format(cam_dir_path, len(frame_contents), num_gt_frames))

        # Split the full video sequence by the containing action sequences
        curr_action: TempAction = None

        # TODO: To support multiple humans merge with gt and add the action sequence to the human (in addition to the action)
        for frame_num, frame_content in frame_contents.items():
            human = get_human_with_highest_score(frame_content)
            gt_row = gt_df[gt_df['frame_num'] == frame_num].iloc[0]
            action_name = gt_row['action']
            if action_name in self.direct_action_mapping:
                action = self.direct_action_mapping[action_name]
            else:
                action = get_action_from_string(action_name)
            if curr_action is None or action_name != curr_action.action_name:
                action_db = HumanAction()
                action_db.action = action.value
                action_db.video_gt = video_gt
                action_db.save()
                curr_action = TempAction(action_name, action_db)

            # Save Frame
            frame_gt = FrameGroundTruth()
            frame_gt.frame_num = frame_num
            frame_gt.video_gt = video_gt
            frame_gt.save()

            if human is not None and human.score >= 0.4:
                # Save Human
                human_db = Human()
                human_db.uid = human.uid
                human_db.frame_gt = frame_gt
                human_db.action = curr_action.action_db
                human_db.datasource = self.datasource_algo.value
                human_db.save()

                # Save Skeleton
                skeleton_joints = get_skeleton_db_from_skeleton(human.skeleton, human_db)
                skeleton_joints.save()

                # Save BB
                bb = get_bb_db_from_skeleton(human.skeleton, human_db)
                bb.save()

    def __transfer_files(self, src_dir: str, dst_dir: str, move_data: bool):
        transfer_files(src=src_dir,
                       dst=dst_dir,
                       move_src_data=move_data)
        batch_rename_files_to_index(dst_dir)

    def __import_video_imgs_data(self, cam_dir_path: str, cam_name: str, seg_src, depth_src, has_segmentation, has_depth, move_data: bool = False):
        img_dir = os.path.join(self.imgs_path, cam_name)

        if os.path.exists(img_dir):
            logger.info("Video images path already exists, skip importing video images.")
        else:
            logger.info('Importing video source data...')
            self.__transfer_files(cam_dir_path, img_dir, move_data)
            if has_segmentation:
                self.__transfer_files(seg_src, os.path.join(self.segmentations_path, cam_name), move_data)
            if has_depth:
                self.__transfer_files(depth_src, os.path.join(self.depths_path, cam_name), move_data)
        # TODO: Segmentation and Depth data needs to be copied as well

    def import_data(self, import_src_files: bool = True):
        for cam_dir_path in get_immediate_subdirectories(self.sim_record_path):
            if cam_dir_path.endswith("annotations"):
                continue
            cam_name = get_last_dir_name(cam_dir_path)
            self.__import_data_for_cam(cam_dir_path, cam_name, import_src_files)

    def __import_data_for_cam(self, cam_dir_path: str, cam_name: str, import_src_files):
        seg_path = os.path.join(cam_dir_path.replace(cam_name, "annotations/seg_{}".format(cam_name)))
        depth_path = os.path.join(cam_dir_path.replace(cam_name, "annotations/depth_{}".format(cam_name)))
        has_segmentation = os.path.exists(seg_path)
        has_depth = os.path.exists(depth_path)
        if import_src_files:
            self.__import_video_imgs_data(cam_dir_path, cam_name, seg_path, depth_path, has_segmentation, has_depth)
        gt_file_path = os.path.join(self.sim_record_path, 'annotations', 'pose3d-view_{}.txt'.format(cam_name))
        if not os.path.exists(gt_file_path):
            raise IOError("GT file not found: '{0}'".format(gt_file_path))
        gt_df = read_annotation_3d(gt_file_path)
        creation_date = datetime.fromtimestamp(os.path.getmtime(self.sim_record_path))
        with cfg.db_conn.atomic() as transaction:  # Opens new transaction.
            try:
                video_gt = VideoGroundTruth()
                video_gt.dataset = self.dataset
                video_gt.vid_img_path = os.path.join(self.imgs_path, cam_name)
                if has_segmentation:
                    video_gt.vid_segmentation_path = seg_path
                if has_depth:
                    video_gt.vid_depth_path = depth_path
                video_gt.vid_name = cam_name
                video_gt.num_frames = gt_df['frame_num'].max() + 1
                video_gt.created_date = creation_date
                video_gt.fps = self.fps
                video_gt.frame_width = self.image_size.width
                video_gt.frame_height = self.image_size.height
                video_gt.save()
                self.dataset_split.video_ground_truths.add(video_gt)

                self.__save_frame_contents_from_pose_rec(cam_dir_path, gt_df, video_gt)
                self.__save_frame_contents_from_gt(gt_df, video_gt)
            except Exception as err:
                print("Error occured, rollback: '{}'".format(err))
                transaction.rollback()
                raise err


if __name__ == "__main__":

    stuff_to_import = ['2019-08-22_ROM01']
    image_size = ImageSize(1280, 720)
    fps = 60
    pose_tracking = True
    direct_action_mapping = {
        "ROM_mcp": Action.WALK,
    }
    for stuff in stuff_to_import:
        importer = CogsysSimImporter("/media/disks/gamma/records/simulation/{}".format(stuff),
                                       "SIM_{}".format(stuff), image_size, fps, pose_tracking, direct_action_mapping,
                                     dataset_part=DatasetPart.TRAIN)
        importer.import_data()
    # TODO: Hangs on Cam change?
