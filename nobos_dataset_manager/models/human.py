from peewee import CharField, ForeignKeyField, FloatField, IntegerField

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.frame_ground_truth import FrameGroundTruth
from nobos_dataset_manager.models.human_action import HumanAction


class Human(BaseModel):
    uid = CharField()
    frame_gt = ForeignKeyField(FrameGroundTruth, backref="humans")
    scale = FloatField(null=True)
    action = ForeignKeyField(HumanAction, backref="humans", null=True, on_delete='CASCADE')
    datasource = IntegerField()
    _bounding_box: 'BoundingBox' = None

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            self._bounding_box = self.bounding_boxes.get()
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value: 'BoundingBox'):
        self._bounding_box = value

    def save(self, *args, **kwargs):
        super(Human, self).save(*args, **kwargs)
        if self._bounding_box is not None:
            self._bounding_box.human = self
            self._bounding_box.save()

    class Meta:
        indexes = (
            (('uid', 'frame_gt', 'datasource'), True),
            (('action', 'frame_gt', 'datasource'), True)
        )
