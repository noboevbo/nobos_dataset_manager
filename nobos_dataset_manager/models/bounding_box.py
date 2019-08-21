from peewee import FloatField, ForeignKeyField

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.human import Human


class BoundingBox(BaseModel):
    top_left_x = FloatField()
    top_left_y = FloatField()
    width = FloatField()
    height = FloatField()
    human = ForeignKeyField(Human, unique=True, null=True, backref="bounding_boxes", on_delete='CASCADE')

