from typing import Dict

from nobos_commons.data_structures.constants.dataset_part import DatasetPart

from nobos_dataset_manager.config import cfg
from nobos_dataset_manager.models.bounding_box import BoundingBox
from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.dataset_split import DatasetSplit, DatasetSplitVideoGroundTruth, get_dataset_splits
from nobos_dataset_manager.models.frame_ground_truth import FrameGroundTruth
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.skeleton_joints import SkeletonJoints
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth

if __name__ == "__main__":
    db = cfg.db_conn
    db.connect()
    db.drop_tables([BoundingBox, Dataset, Human, HumanAction, FrameGroundTruth, SkeletonJoints, VideoGroundTruth])
    db.create_tables([BoundingBox, Dataset, Human, HumanAction, FrameGroundTruth, SkeletonJoints, VideoGroundTruth])

    db.create_tables([
        DatasetSplit,
        DatasetSplitVideoGroundTruth])

    with db.atomic() as transaction:  # Opens new transaction.
        try:
            for dataset in Dataset.select():
                video_gts = VideoGroundTruth.select().where(
                    VideoGroundTruth.dataset == dataset.id
                )
                splits: Dict[DatasetPart, DatasetSplit] = get_dataset_splits("JHMDB")
                for video_gt in video_gts:
                    splits[DatasetPart(video_gt.dataset_part)].video_ground_truths.add(video_gt)
        except Exception as err:
            # Because this block of code is wrapped with "atomic", a
            # new transaction will begin automatically after the call
            # to rollback().
            print(err)
            transaction.rollback()
