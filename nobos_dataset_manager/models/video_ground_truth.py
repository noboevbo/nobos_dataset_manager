from peewee import ForeignKeyField, CharField, IntegerField, DateTimeField, Check

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.dataset import Dataset


class VideoGroundTruth(BaseModel):
    dataset = ForeignKeyField(Dataset, backref='video_ground_truths', on_delete='CASCADE')

    vid_path = CharField(unique=True, null=True)
    vid_img_path = CharField(unique=True, null=True)
    vid_name = CharField()
    fps = IntegerField()
    num_frames = IntegerField()
    frame_width = IntegerField()
    frame_height = IntegerField()
    viewport = CharField(null=True)
    created_date = DateTimeField(null=True)

    class Meta:
        constraints = [Check('vid_path IS NOT NULL or vid_img_path IS NOT NULL')]

    # def __repr__(self):
    #     return "VideoGroundTruth: (ID='{0}', Dataset='{1}', vid_path='{2}', vid_name='{3}', num_frames='{4}', viewport='{5}', date_created='{6}')".format(
    #         self.id, self.dataset.name, self.vid_path, self.vid_name, self.num_frames, self.viewport, self.date_created)
