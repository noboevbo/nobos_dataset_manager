from peewee import IntegerField, ForeignKeyField

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth


class HumanAction(BaseModel):
    action = IntegerField(null=False)
    video_gt = ForeignKeyField(VideoGroundTruth, backref='actions', on_delete='CASCADE')
