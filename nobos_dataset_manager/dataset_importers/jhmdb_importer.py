import datetime
import os
from time import strptime, strftime, mktime
from typing import List, Dict, Any

import numpy as np
from nobos_commons.data_structures.constants.cardinal_point import CardinalPoint, CardinalPointAbbreviation
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.data_structures.image_content import ImageContent
from nobos_commons.data_structures.skeletons.skeleton_jhmdb import SkeletonJhmdb
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.input_providers.camera.img_dir_provider import ImgDirProvider
from nobos_commons.tools.file_manager import transfer_files
from nobos_commons.tools.log_handler import logger
from nobos_commons.tools.skeleton_converters.skeleton_converter_factory import SkeletonConverterFactory
from nobos_commons.tools.skeleton_converters.skeleton_converter_jhmdb_to_stickman import \
    SkeletonConverterJhmdbToStickman
from nobos_commons.utils.file_helper import get_immediate_subdirectories, batch_rename_files_to_index
from scipy.io import loadmat

from nobos_dataset_manager import configurator
from nobos_dataset_manager.config import cfg
from nobos_dataset_manager.dataset_tools.pipeline_ehpi import PipelineEhpi
from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.dataset_split import DatasetSplit, get_dataset_splits
from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.frame_ground_truth import FrameGroundTruth
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth
from nobos_dataset_manager.utils import get_skeleton_db_from_skeleton, get_bb_db_from_skeleton


def get_jhmdb_dataset_parts(splits_dir_path: str, split_version: int, split_pattern: str = 'split{}.txt'):
    jhmdb_dataset_parts: Dict[str, DatasetPart] = {}
    for file_name in os.listdir(splits_dir_path):
        assert file_name.endswith('.txt')
        if not file_name.endswith(split_pattern.format(split_version)):  # For now only import split 1, 2 and 3 are updated later
            continue
        print(file_name)
        with open(os.path.join(splits_dir_path, file_name), 'r') as split_file:
            split_list = split_file.read().split('\n')
            for split in split_list:
                if not split:  # empty line
                    continue
                video_name, set_num = split.split(' ')
                dataset_part = DatasetPart.TRAIN
                if int(set_num) == 2:
                    dataset_part = DatasetPart.TEST
                jhmdb_dataset_parts[video_name] = dataset_part
    return jhmdb_dataset_parts

class JhmdbImporter(object):
    def __init__(self, dataset_path: str, skeleton_converter: SkeletonConverterJhmdbToStickman,
                 datasource: Datasource, img_dir_to_skeleton_tool, is_additional_source: bool = False):
        """
        Importer for a JHMDB Dataset. Dataset source: http://jhmdb.is.tue.mpg.de/
        :param dataset_path: The path to the JHMDB root folder
        """
        assert os.path.exists(dataset_path), "Dataset could not be found!"
        self.dataset_path: str = dataset_path
        self.dataset, created = Dataset.get_or_create(name="JHMDB")
        self.num_joints: str = len(SkeletonJhmdb.joints)
        self.path_rename_mapping: Dict[str, str] = {}
        self.skeleton_converter = skeleton_converter
        self.image_size = ImageSize(320, 240)
        self.img_blob_path = os.path.join('data', 'video_gt', 'JHMDB', 'imgs')
        self.vid_blob_path = os.path.join('data', 'video_gt', 'JHMDB', 'vids')
        self.imgs_path = os.path.join(cfg.blob_storage_path, self.img_blob_path)
        self.vids_path = os.path.join(cfg.blob_storage_path, self.vid_blob_path)

        self.datasource = datasource
        self.is_additional_source = is_additional_source
        self.img_dir_to_skeleton_tool = img_dir_to_skeleton_tool

    def import_data(self, move_data: bool = False):
        """
        Expects JHMDB extracted in a folder, e.g. root -> [joint_positions, puppet_flow_ann, puppet_mask,
        ReCompress_Videos, [...]].
        - joint order JHMDB:  Neck, Belly, Head, R_Shoulder, L_Shoulder, R_Hip, L_Hip, R_Elbow, L_Elbow, R_Knee, L_Knee,
          R_Wrist, L_Wrist, R_Ankle, L_Ankle
        - scale JHMDB: TODO 1 is the base size of the puppet, so how to convert this in standard human scale? What is their base puppet size?
        It is estimated 90% of image height from head to foot joint. Independent of invisible joints
        - Viewpoint: Direction of the person: S, SSW, SW, WSW, W, WNW, NW, NNW, N, NNE, NE, ENE, E, ESE, SE, SSE
          where S is towards the camera and N away from camera.
        :param move_data: Specifies whether the src files should be moved, if not they will be copied
        :return:
        """

        self.__import_video_data(move_data)
        self.__import_video_imgs_data(move_data)

        logger.info('Importing dataset split...')
        splits_dir_path = os.path.join(self.dataset_path, 'splits')
        jhmdb_dataset_parts: Dict[str, DatasetPart] = get_jhmdb_dataset_parts(splits_dir_path, split_version=1)

        logger.info('Importing ground truth data...')
        self.import_ground_truth_to_db(jhmdb_dataset_parts)
        # Todo: Automated eye and ear by head and neck position as well as viewpoint and scale


    def __import_video_data(self, move_data: bool = False):
        if os.path.exists(self.vids_path):
            logger.info("Videos path already exists, skip importing videos.")
        else:
            dataset_dir = os.path.join(self.dataset_path, 'ReCompress_Videos')
            for action_video_path in os.listdir(dataset_dir):
                if action_video_path.startswith("."):
                    continue
                logger.info('Importing video source data...')
                transfer_files(src=os.path.join(dataset_dir, action_video_path),
                               dst=self.vids_path,
                               move_src_data=move_data)

    def __import_video_imgs_data(self, move_data: bool = False):
        if os.path.exists(self.imgs_path):
            logger.info("Video images path already exists, skip importing video images.")
        else:
            logger.info('Importing video source data...')
            dataset_dir = os.path.join(self.dataset_path, 'Rename_Images')
            for action_video_path in os.listdir(dataset_dir):
                if action_video_path.startswith("."):
                    continue
                transfer_files(src=os.path.join(dataset_dir, action_video_path),
                               dst=self.imgs_path,
                               move_src_data=move_data)
            for action_img_dir in get_immediate_subdirectories(self.imgs_path):
                batch_rename_files_to_index(action_img_dir)

    def import_ground_truth_to_db(self, jhmdb_dataset_parts: Dict[str, DatasetPart]):
        for root, dirs, files in os.walk(os.path.join(self.dataset_path, 'joint_positions')):
            if len(files) == 1 and files[0] == 'joint_positions.mat':
                parent_dirs = root.split(os.sep)
                unique_id = parent_dirs[-1]
                action_name = parent_dirs[-2]
                vid_filename = '{0}.avi'.format(unique_id)
                vid_path = os.path.join(self.vid_blob_path, vid_filename)
                vid_img_path = os.path.join(self.img_blob_path, unique_id)
                dataset_part = jhmdb_dataset_parts[vid_filename]
                logger.info('Import GT data for vid_path {}'.format(vid_path))
                # assert os.path.exists(vid_path), 'Error: No source video could be found at {0}'.format(vid_path)
                action = Action[action_name.upper()]
                self.save_gt(joint_positions_mat_path=os.path.join(root, files[0]), vid_path=vid_path, vid_img_path=vid_img_path,
                             unique_id=vid_filename, dataset_part=dataset_part, action=action)

    def __get_frame_contents(self, vid_img_dir) -> Dict[int, ImageContent]:
        # TODO: Not run on src image dir, run on blob dir
        img_dir_provider = ImgDirProvider(vid_img_dir, image_size=self.image_size, print_current_path=True)
        return self.img_dir_to_skeleton_tool.get_from_input_provider(img_dir_provider)

    def save_gt(self, joint_positions_mat_path: str, vid_path: str, vid_img_path: str, unique_id, dataset_part: DatasetPart, action: Action,
                ):
        joint_positions_mat = loadmat(joint_positions_mat_path)

        number_of_frames = self.__get_number_of_frames(joint_positions_mat)
        scale_per_frame = self.__get_scale_per_frame(joint_positions_mat)
        viewpoint = self.__get_viewpoint(joint_positions_mat)
        creation_date = self.__get_creation_date(joint_positions_mat)

        if self.datasource is Datasource.GROUND_TRUTH:
            joint_positions = self.__get_joint_positions(joint_positions_mat)
            joint_positions_world = self.__get_joint_positions_world(joint_positions_mat)
        else:
            joint_positions = self.__get_frame_contents(os.path.join(self.dataset_path, "Rename_Images", action.name.lower(), unique_id.replace('.avi', '')))
            joint_positions_world = None

        video_gt = VideoGroundTruth()
        if self.is_additional_source:
            video_gt, created = video_gt.get_or_create(dataset=self.dataset,
                                                       vid_path=vid_path,
                                                       vid_img_path=vid_img_path,
                                                       vid_name=unique_id,
                                                       num_frames=number_of_frames,
                                                       fps=30,
                                                       frame_width=self.image_size.width,
                                                       frame_height=self.image_size.height
                                                       )
            if created:
                raise Exception("This VIDEO GT does not exist, no additional source..")
        else:
            video_gt.dataset = self.dataset
            video_gt.vid_path = vid_path
            video_gt.vid_img_path = vid_img_path
            video_gt.fps = 30
            video_gt.vid_name = unique_id
            video_gt.num_frames = number_of_frames
            video_gt.viewport = viewpoint
            video_gt.created_date = creation_date
            video_gt.frame_width = self.image_size.width
            video_gt.frame_height = self.image_size.height
            video_gt.save()

        # The full sequence is labeled as one action
        action_db = HumanAction()
        action_db.action = action.value
        action_db.video_gt = video_gt
        action_db.save()

        self.save_frames_gt(video_gt, number_of_frames, scale_per_frame, joint_positions, joint_positions_world, action=action_db)

    def save_frames_gt(self, video_gt: VideoGroundTruth, number_of_frames: int, scales: List[float], joint_positions: np.ndarray,
                               joint_positions_world: np.ndarray, action: HumanAction):
        last_human_uid = None
        for frame_num in range(0, number_of_frames):
            frame_gt = FrameGroundTruth()
            if self.is_additional_source:
                frame_gt, created = frame_gt.get_or_create(frame_num=frame_num, video_gt=video_gt)
                if created:
                    raise Exception("This Frame GT does not exist, no additional source..")
            else:
                frame_gt.frame_num = frame_num
                frame_gt.video_gt = video_gt
                frame_gt.save()

            joint_positions_frame = joint_positions[frame_num]
            human = Human()
            human.uid = "00000"  # Only 1 human per frame in JHMDB
            human.scale = scales[frame_num]
            human.frame_gt = frame_gt
            human.action = action
            human.datasource = self.datasource.value
            if self.datasource == Datasource.GROUND_TRUTH:
                # joint_positions_world_frame = joint_positions_world[frame_num]
                skeleton_jhmdb = self.__get_skeleton_jhmdb(joint_positions_frame)

                skeleton = self.skeleton_converter.get_converted_skeleton(skeleton_jhmdb)
            else:
                image_content = joint_positions_frame
                if len(image_content.humans) > 1:
                    a = 1
                human_to_use = None
                for human_pred in image_content.humans:
                    if last_human_uid is not None and last_human_uid == human_pred.uid:
                        human_to_use = human_pred
                        break
                    elif human_to_use is None or human_to_use.score < human_pred.score:
                        human_to_use = human_pred

                if human_to_use is None:
                    continue
                if last_human_uid is not None and human_to_use.uid != last_human_uid:
                    a = 1

                last_human_uid = human_to_use.uid
                human.uid = human_to_use.uid
                skeleton = human_to_use.skeleton

            human.save()
            skeleton_joints = get_skeleton_db_from_skeleton(skeleton, human)
            skeleton_joints.save()

            bb = get_bb_db_from_skeleton(skeleton, human)
            bb.save()


    def __get_skeleton_jhmdb(self, joint_positions):
        skeleton_jhmdb = SkeletonJhmdb()
        for joint_num, joint in enumerate(skeleton_jhmdb.joints):
            joint.x = joint_positions[joint_num][0]
            joint.y = joint_positions[joint_num][1]
            joint.score = 1.0
        return skeleton_jhmdb

    @staticmethod
    def __get_number_of_frames(joint_positions_mat: Dict[str, Any]) -> int:
        num_frames = len(joint_positions_mat['scale'][0])
        logger.info('Extracted number of frames: {}'.format(num_frames))
        return num_frames

    @staticmethod
    def __get_scale_per_frame(joint_positions_mat: Dict[str, Any]) -> List[float]:
        scale_per_frame = joint_positions_mat['scale'][0]
        return scale_per_frame

    @staticmethod
    def __get_viewpoint(joint_positions_mat: Dict[str, Any]) -> CardinalPoint:
        assert len(joint_positions_mat['viewpoint']) == 1, 'More than one viewpoint found, unhandled.'
        viewpoint = CardinalPointAbbreviation[joint_positions_mat['viewpoint'][0]].value
        logger.info('Extracted viewpoint: {}'.format(viewpoint.name))
        return viewpoint

    @staticmethod
    def __get_creation_date(joint_positions_mat: Dict[str, Any]) -> datetime.datetime:
        header = joint_positions_mat['__header__'].decode("utf-8")
        created_on = header[-20:]
        logger.debug('Try parsing header creation date: {}'.format(created_on))
        creation_date = strptime(created_on, '%b %d %H:%M:%S %Y')
        logger.info('Extracted creation date: {}'.format(strftime("%Y-%m-%d %H:%M:%S", creation_date)))
        creation_date = datetime.datetime.fromtimestamp(mktime(creation_date))
        return creation_date

    @staticmethod
    def __get_joint_positions(joint_positions_mat: Dict[str, Any]) -> np.ndarray:
        joint_positions = joint_positions_mat['pos_img'].transpose(2, 1, 0)
        logger.info('Extracted joint positions with shape: {}'.format(joint_positions.shape))
        return joint_positions

    @staticmethod
    def __get_joint_positions_world(joint_positions_mat: Dict[str, Any]) -> np.ndarray:
        joint_positions_world = joint_positions_mat['pos_world'].transpose(2, 1, 0)
        logger.info('Extracted joint positions (world) with shape: {}'.format(joint_positions_world.shape))
        return joint_positions_world
#
def set_splits_for_video_gt(splits_dir_path: str, split_version: int, is_subsplit: bool = False):
    split_pattern = 'split{}.txt'
    if is_subsplit:
        split_pattern = 'split_{}.txt'
    jhmdb_dataset_parts: Dict[str, DatasetPart] = get_jhmdb_dataset_parts(splits_dir_path, split_version=split_version, split_pattern=split_pattern)
    dataset, created = Dataset.get_or_create(name="JHMDB")
    if created:
        raise Exception("JHMDB NOT EXISTS!!")
    with cfg.db_conn.atomic() as transaction:  # Opens new transaction.
        try:
            video_gts = VideoGroundTruth.select().where(
                VideoGroundTruth.dataset == dataset.id
            )
            split_name = "JHMDB-{}".format(split_version)
            if is_subsplit:
                split_name = "SUB-" + split_name
            splits: Dict[DatasetPart, DatasetSplit] = get_dataset_splits(split_name)
            for video_gt in video_gts:
                if is_subsplit and video_gt.vid_name not in jhmdb_dataset_parts:
                    continue
                split = jhmdb_dataset_parts[video_gt.vid_name]
                splits[split].video_ground_truths.add(video_gt)
        except Exception as err:
            print(err)
            transaction.rollback()
