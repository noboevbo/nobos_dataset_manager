import datetime

from peewee import DateTimeField, Model

from nobos_dataset_manager.config import cfg


class BaseModel(Model):
    class Meta:
        database = cfg.db_conn

    date_created = DateTimeField(default=datetime.datetime.now)


class BaseViewModel(Model):
    class Meta:
        database = cfg.db_conn
        primary_key = False
