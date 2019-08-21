from typing import Dict

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from peewee import CharField, ManyToManyField, IntegerField

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth


class DatasetSplit(BaseModel):
    video_ground_truths = ManyToManyField(VideoGroundTruth,
                                          backref='dataset_splits',
                                          on_delete='CASCADE')
    name = CharField()
    dataset_part = IntegerField()

    class Meta:
        indexes = (
            (('name', 'dataset_part'), True),
        )

DatasetSplitVideoGroundTruth = DatasetSplit.video_ground_truths.get_through_model()


def get_dataset_splits(dataset_split_name: str) -> Dict[DatasetPart, DatasetSplit]:
    splits: Dict[DatasetPart, DatasetSplit] = {}
    splits[DatasetPart.TRAIN], created = DatasetSplit.get_or_create(name=dataset_split_name,
                                                                    dataset_part=DatasetPart.TRAIN.value)
    splits[DatasetPart.TEST], created = DatasetSplit.get_or_create(name=dataset_split_name,
                                                                   dataset_part=DatasetPart.TEST.value)
    splits[DatasetPart.VALIDATION], created = DatasetSplit.get_or_create(name=dataset_split_name,
                                                                         dataset_part=DatasetPart.VALIDATION.value)
    return splits

