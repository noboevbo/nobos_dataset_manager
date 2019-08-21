from peewee import CharField

from nobos_dataset_manager.models.base_model import BaseModel


class Dataset(BaseModel):
    name = CharField(unique=True)
