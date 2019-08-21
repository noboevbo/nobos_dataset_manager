import os
from datetime import datetime
from shutil import copy2, move
from typing import List, Dict

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.input_providers.camera.img_dir_provider import ImgDirProvider
from nobos_commons.tools.file_manager import transfer_files
from nobos_commons.tools.log_handler import logger
from nobos_commons.utils.file_helper import get_filename_from_path, get_immediate_subdirectories, get_last_dir_name, \
    get_create_path, batch_rename_files_to_index
from peewee import DoesNotExist

from nobos_dataset_manager import configurator
from nobos_dataset_manager.config import cfg
from nobos_dataset_manager.dataset_tools.pipeline_ehpi import PipelineEhpi
from nobos_dataset_manager.models.base_model import db
from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.dataset_split import DatasetSplit, get_dataset_splits
from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.frame_ground_truth import FrameGroundTruth
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth
from nobos_dataset_manager.utils import get_skeleton_db_from_skeleton, get_bb_db_from_skeleton


# Webcam Records

class ImageFolderImportData(object):
    def __init__(self, img_dir: str):
        self.img_dir: str = img_dir
        self.vid_path: str = None
        self.action: Action = None


def __get_vid_file_name(webcam_record_path: str, webcam_record_name: str, vid_ext: str):
    vid_file_name = "{0}{1}".format(webcam_record_name, vid_ext)
    vid_file_path = os.path.join(webcam_record_path, vid_file_name)
    if not os.path.exists(vid_file_path):
        vid_file_path = None
    return vid_file_path


def get_info_from_webcam_records(webcam_record_path: str, vid_ext=".avi"):
    webcam_records: List[ImageFolderImportData] = []
    for sub_dir in get_immediate_subdirectories(webcam_record_path):
        data = ImageFolderImportData(img_dir=sub_dir)  # TODO Os path join webcam rec path?
        webcam_record_name = get_last_dir_name(sub_dir)
        data.vid_path = __get_vid_file_name(webcam_record_path, webcam_record_name, vid_ext)
        action_name = webcam_record_name.rsplit("_", maxsplit=1)[0].upper()
        if action_name not in Action.__members__:
            print("Action with name: {0} does not exist for path: {1} -> SKIP!".format(action_name, sub_dir))
            continue
        data.action = Action[action_name.upper()]
        webcam_records.append(data)
    return webcam_records


class ImageFolderImporter(object):
    def __init__(self, import_data: ImageFolderImportData, dataset_name: str, image_size: ImageSize, fps: int,
                 pose_tracking: bool, dataset_part: DatasetPart, datasource: Datasource, img_dir_to_skeleton_tool,
                 is_additional_source: bool = False,
                 ):
        """
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
        assert os.path.exists(import_data.img_dir), "image folder '{}' could not be found!".format(import_data.img_dir)

        self.is_additional_source = is_additional_source
        self.img_dir_to_skeleton_tool = img_dir_to_skeleton_tool
        self.pose_tracking = pose_tracking
        self.skip = False
        self.dataset, created = Dataset.get_or_create(name=dataset_name)
        self.dataset_splits: Dict[DatasetPart, DatasetSplit] = get_dataset_splits('{}-1'.format(dataset_name))
        self.dataset_name = dataset_name
        self.src_image_dir: str = import_data.img_dir
        self.src_vid_path = import_data.vid_path
        self.fps = fps
        self.image_size = image_size
        self.action = import_data.action
        self.datasource = datasource

        self.img_blob_path = os.path.join('data', 'video_gt', dataset_name, 'imgs')
        self.vid_blob_path = os.path.join('data', 'video_gt', dataset_name, 'vids')
        self.imgs_path = get_create_path(os.path.join(cfg.blob_storage_path, self.img_blob_path))
        self.vids_path = get_create_path(os.path.join(cfg.blob_storage_path, self.vid_blob_path))

        self.unique_id = get_last_dir_name(os.path.basename(os.path.normpath(self.src_image_dir)))
        try:
            videogt = VideoGroundTruth.get(vid_img_path=os.path.join(self.imgs_path, self.unique_id))
            print("EXISTS!")
            self.skip = True
        except DoesNotExist:
            print("Does not exist, will create.")
        self.dataset_part = dataset_part

    def __get_frame_contents(self) -> Dict[int, ImageContent]:
        # TODO: Not run on src image dir, run on blob dir
        img_dir_provider = ImgDirProvider(self.src_image_dir, image_size=self.image_size, print_current_path=True)
        return self.img_dir_to_skeleton_tool.get_frame_contents(img_dir_provider)

    def __import_video_data(self, move_data: bool = False):
        if self.src_vid_path is None:
            return
        logger.info('Importing video source data...')
        if move_data:
            move(self.src_vid_path, self.vids_path)
        else:
            copy2(self.src_vid_path, self.vids_path)

    def __import_video_imgs_data(self, move_data: bool = False):
        logger.info('Importing video source data...')
        img_dir = os.path.join(self.imgs_path, self.unique_id)
        transfer_files(src=os.path.join(self.src_image_dir),
                       dst=img_dir,
                       move_src_data=move_data)
        batch_rename_files_to_index(img_dir)

    def import_data(self, import_src_files: bool = True):
        if self.skip:
            return
        if import_src_files:
            self.__import_video_data()
            self.__import_video_imgs_data()
        frame_contents = self.__get_frame_contents()
        creation_date = datetime.fromtimestamp(os.path.getmtime(self.src_image_dir))
        with db.atomic() as transaction:  # Opens new transaction.
            try:
                video_gt = VideoGroundTruth()
                vid_path = os.path.join(self.vid_blob_path, get_filename_from_path(self.src_vid_path))
                vid_img_path = os.path.join(self.img_blob_path, self.unique_id)
                if self.is_additional_source:
                    video_gt, created = video_gt.get_or_create(dataset=self.dataset,
                                                               vid_path=vid_path,
                                                               vid_img_path=vid_img_path,
                                                               vid_name=self.unique_id,
                                                               num_frames=len(frame_contents),
                                                               dataset_part=self.dataset_part.value,
                                                               fps=self.fps,
                                                               frame_width=self.image_size.width,
                                                               frame_height=self.image_size.height
                                                               )
                    if created:
                        raise Exception("This VIDEO GT does not exist, no additional source..")
                else:
                    video_gt.dataset = self.dataset
                    video_gt.vid_path = vid_path
                    video_gt.vid_img_path = vid_img_path
                    video_gt.vid_name = self.unique_id
                    video_gt.num_frames = len(frame_contents)
                    video_gt.created_date = creation_date
                    video_gt.dataset_part = self.dataset_part.value
                    video_gt.fps = self.fps
                    video_gt.frame_width = self.image_size.width
                    video_gt.frame_height = self.image_size.height
                    video_gt.save()

                    self.dataset_splits[DatasetPart(self.dataset_part.value)].video_ground_truths.add(video_gt)

                # TODO
                # The full sequence is labeled as one action
                action_db = HumanAction()
                action_db.action = self.action.value
                action_db.video_gt = video_gt
                action_db.save()
                self.__save_skeleton_joints(video_gt, action_db, frame_contents)
            except Exception as err:
                logger.error(err)
                transaction.rollback()

    def __save_skeleton_joints(self, video_gt: VideoGroundTruth, action_db: HumanAction,
                               frame_contents: Dict[int, ImageContent]):
        for frame_id, image_content in frame_contents.items():
            frame_gt = FrameGroundTruth()
            if self.is_additional_source:
                frame_gt, created = frame_gt.get_or_create(frame_num=frame_id, video_gt=video_gt)
                if created:
                    raise Exception("This Frame GT does not exist, no additional source..")
            else:
                frame_gt.frame_num = frame_id
                frame_gt.video_gt = video_gt
                frame_gt.save()
            # TODO: Currently a hack to only use one human, fix when multiple human support is required
            human_to_use = None
            for human in image_content.humans:
                if human_to_use is None or human_to_use.score < human.score:
                    human_to_use = human
            if human_to_use is None:
                continue
            if len(image_content.humans) > 1:
                a = 1
            # for human in image_content.humans:
            human_db = Human()
            human_db.uid = human_to_use.uid  # Only 1 human per frame in JHMDB
            human_db.frame_gt = frame_gt
            human_db.action = action_db
            human_db.datasource = self.datasource.value
            human_db.save()

            skeleton_joints = get_skeleton_db_from_skeleton(human_to_use.skeleton, human_db)
            skeleton_joints.save()

            bb = get_bb_db_from_skeleton(human_to_use.skeleton, human_db)
            bb.save()


def standard_import():
    stuff_to_import = [
        "2019_03_11_OFP_Records_Yi",
        "2019_03_11_OFP_Records_HSRT_Yi_HARD",
        "2019_03_11_OFP_Records_HSRT_Yi",
        "2019_03_11_OFP_Records_HSRT_Dashcam",
        "2019_03_11_OFP_Records_Dashcam",
        "2019_03_11_OFP_Hella_Rec01",
        "2019_02_27_Hella_OFP_ParkingLot",
    ]
    # stuff_to_import = ["VICON_2019_03_06_Videos_Vue01"]
    configurator.setup()
    image_size = ImageSize(1280, 720)
    fps = 30
    pose_tracking = True
    for stuff in stuff_to_import:
        webcam_folders = get_info_from_webcam_records(
            "/media/disks/beta/nobos_dataset_manager/data/video_gt/{}".format(stuff), vid_ext=".mp4")
        # test_import = []
        # for folder in webcam_folders:
        #     if folder.img_dir == "/media/disks/beta/tmp_vicon/VICON_2019_03_06_Videos_Vue01/walk_14.2112013":
        #         test_import.append(folder)
        for webcam_record in webcam_folders:
            importer = ImageFolderImporter(webcam_record, stuff, image_size, fps, pose_tracking,
                                           dataset_part=DatasetPart.TRAIN,
                                           datasource=Datasource.OFP_POSE_RECOGNITION)
            importer.import_data()


def import_additional_hella():
    stuff_to_import = [
        # "2019_03_11_OFP_Records_Yi",
        "2019_03_11_OFP_Records_HSRT_Yi_HARD",
        "2019_03_11_OFP_Records_HSRT_Yi",
        "2019_03_11_OFP_Records_HSRT_Dashcam",
        # "2019_03_11_OFP_Records_Dashcam",
        # "2019_03_11_OFP_Hella_Rec01",
        # "2019_02_27_Hella_OFP_ParkingLot",
    ]
    # stuff_to_import = ["VICON_2019_03_06_Videos_Vue01"]
    configurator.setup()
    image_size = ImageSize(1280, 720)
    fps = 30
    pose_tracking = True
    for stuff in stuff_to_import:
        webcam_folders = get_info_from_webcam_records(
            "/media/disks/gamma/nobos_dataset_manager/data/video_gt/{}/imgs".format(stuff), vid_ext=".mp4")
        # test_import = []
        # for folder in webcam_folders:
        #     if folder.img_dir == "/media/disks/beta/tmp_vicon/VICON_2019_03_06_Videos_Vue01/walk_14.2112013":
        #         test_import.append(folder)
        for webcam_record in webcam_folders:
            importer = ImageFolderImporter(webcam_record, stuff, image_size, fps, pose_tracking,
                                           dataset_part=DatasetPart.TEST,
                                           datasource=Datasource.OFP_POSE_RECOGNITION_FAST_MODE,
                                           img_dir_to_skeleton_tool=PipelineEhpi(use_pose_tracking=pose_tracking, image_size=image_size, use_fast_mode=True),
                                           is_additional_source=True)
            importer.import_data(False)


def import_2019_03_13_train(use_fast_mode: bool):
    stuff_to_import = [
        "2019_03_13_Freilichtmuseum_Dashcam_01",
        "2019_03_13_Freilichtmuseum_Yi_01",
    ]
    # stuff_to_import = ["VICON_2019_03_06_Videos_Vue01"]
    configurator.setup()
    image_size = ImageSize(1280, 720)
    fps = 30
    pose_tracking = True
    for stuff in stuff_to_import:
        webcam_folders = get_info_from_webcam_records(
            "/media/disks/beta/records/real_cam/{}".format(stuff), vid_ext=".mp4")
        # test_import = []
        # for folder in webcam_folders:
        #     if folder.img_dir == "/media/disks/beta/tmp_vicon/VICON_2019_03_06_Videos_Vue01/walk_14.2112013":
        #         test_import.append(folder)
        for webcam_record in webcam_folders:
            datasource = Datasource.OFP_POSE_RECOGNITION_FAST_MODE if use_fast_mode else Datasource.OFP_POSE_RECOGNITION
            importer = ImageFolderImporter(webcam_record, stuff, image_size, fps, pose_tracking,
                                           dataset_part=DatasetPart.TRAIN,
                                           datasource=datasource, is_additional_source=use_fast_mode,
                                           img_dir_to_skeleton_tool=PipelineEhpi(use_pose_tracking=pose_tracking, image_size=image_size, use_fast_mode=True))
            importer.import_data(not use_fast_mode)


def import_2019_03_13_test(use_fast_mode: bool):
    stuff_to_import = [
        "2019_03_13_Freilichtmuseum_Dashcam_02",
        "2019_03_13_Freilichtmuseum_Yi_02",
    ]
    # stuff_to_import = ["VICON_2019_03_06_Videos_Vue01"]
    configurator.setup()
    image_size = ImageSize(1280, 720)
    fps = 30
    pose_tracking = True
    for stuff in stuff_to_import:
        webcam_folders = get_info_from_webcam_records(
            "/media/disks/beta/records/real_cam/{}".format(stuff), vid_ext=".mp4")
        # test_import = []
        # for folder in webcam_folders:
        #     if folder.img_dir == "/media/disks/beta/tmp_vicon/VICON_2019_03_06_Videos_Vue01/walk_14.2112013":
        #         test_import.append(folder)
        for webcam_record in webcam_folders:
            datasource = Datasource.OFP_POSE_RECOGNITION_FAST_MODE if use_fast_mode else Datasource.OFP_POSE_RECOGNITION
            importer = ImageFolderImporter(webcam_record, stuff, image_size, fps, pose_tracking,
                                           dataset_part=DatasetPart.TEST,
                                           datasource=datasource, is_additional_source=use_fast_mode,
                                           img_dir_to_skeleton_tool=PipelineEhpi(use_pose_tracking=pose_tracking, image_size=image_size, use_fast_mode=True))
            importer.import_data(not use_fast_mode)

def import_2019_04_10_journal_eval(use_fast_mode: bool):
    stuff_to_import = [
        "2019_ITS_Journal_Eval2",
    ]
    # stuff_to_import = ["VICON_2019_03_06_Videos_Vue01"]
    configurator.setup()
    image_size = ImageSize(1280, 720)
    fps = 30
    pose_tracking = True
    for stuff in stuff_to_import:
        webcam_folders = get_info_from_webcam_records(
            "/media/disks/beta/records/webcam/{}".format(stuff), vid_ext=".avi")
        # test_import = []
        # for folder in webcam_folders:
        #     if folder.img_dir == "/media/disks/beta/tmp_vicon/VICON_2019_03_06_Videos_Vue01/walk_14.2112013":
        #         test_import.append(folder)
        for webcam_record in webcam_folders:
            datasource = Datasource.OFP_POSE_RECOGNITION_FAST_MODE if use_fast_mode else Datasource.OFP_POSE_RECOGNITION
            importer = ImageFolderImporter(webcam_record, stuff, image_size, fps, pose_tracking,
                                           dataset_part=DatasetPart.TEST,
                                           datasource=datasource, is_additional_source=use_fast_mode,
                                           img_dir_to_skeleton_tool=PipelineEhpi(use_pose_tracking=pose_tracking, image_size=image_size, use_fast_mode=True))
            importer.import_data(not use_fast_mode)


if __name__ == "__main__":
    import_2019_04_10_journal_eval(False)
    # import_2019_03_13_train(False)
    # import_2019_03_13_test(False)
    # import_2019_03_13_train(True)
    # import_2019_03_13_test(True)
