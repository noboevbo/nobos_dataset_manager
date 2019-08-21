from peewee import ForeignKeyField, IntegerField

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth


class FrameGroundTruth(BaseModel):
    video_gt = ForeignKeyField(VideoGroundTruth, backref='frames', on_delete='CASCADE')
    frame_num = IntegerField()

    class Meta:
        indexes = (
            (('video_gt', 'frame_num'), True),
        )
