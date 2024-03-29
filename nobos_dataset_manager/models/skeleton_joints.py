from peewee import FloatField, IntegerField, ForeignKeyField

from nobos_dataset_manager.models.base_model import BaseModel
from nobos_dataset_manager.models.human import Human


class SkeletonJoints(BaseModel):
    human = ForeignKeyField(Human, backref='joints', unique=True, on_delete='CASCADE')
    nose_x = FloatField(null=True)
    nose_y = FloatField(null=True)
    nose_z = FloatField(null=True)
    nose_score = FloatField(null=True)
    nose_is_visible = IntegerField(null=True)

    neck_x = FloatField(null=True)
    neck_y = FloatField(null=True)
    neck_z = FloatField(null=True)
    neck_score = FloatField(null=True)
    neck_is_visible = IntegerField(null=True)

    right_shoulder_x = FloatField(null=True)
    right_shoulder_y = FloatField(null=True)
    right_shoulder_z = FloatField(null=True)
    right_shoulder_score = FloatField(null=True)
    right_shoulder_is_visible = IntegerField(null=True)

    right_elbow_x = FloatField(null=True)
    right_elbow_y = FloatField(null=True)
    right_elbow_z = FloatField(null=True)
    right_elbow_score = FloatField(null=True)
    right_elbow_is_visible = IntegerField(null=True)

    right_wrist_x = FloatField(null=True)
    right_wrist_y = FloatField(null=True)
    right_wrist_z = FloatField(null=True)
    right_wrist_score = FloatField(null=True)
    right_wrist_is_visible = IntegerField(null=True)

    left_shoulder_x = FloatField(null=True)
    left_shoulder_y = FloatField(null=True)
    left_shoulder_z = FloatField(null=True)
    left_shoulder_score = FloatField(null=True)
    left_shoulder_is_visible = IntegerField(null=True)

    left_elbow_x = FloatField(null=True)
    left_elbow_y = FloatField(null=True)
    left_elbow_z = FloatField(null=True)
    left_elbow_score = FloatField(null=True)
    left_elbow_is_visible = IntegerField(null=True)

    left_wrist_x = FloatField(null=True)
    left_wrist_y = FloatField(null=True)
    left_wrist_z = FloatField(null=True)
    left_wrist_score = FloatField(null=True)
    left_wrist_is_visible = IntegerField(null=True)

    right_hip_x = FloatField(null=True)
    right_hip_y = FloatField(null=True)
    right_hip_z = FloatField(null=True)
    right_hip_score = FloatField(null=True)
    right_hip_is_visible = IntegerField(null=True)

    right_knee_x = FloatField(null=True)
    right_knee_y = FloatField(null=True)
    right_knee_z = FloatField(null=True)
    right_knee_score = FloatField(null=True)
    right_knee_is_visible = IntegerField(null=True)

    right_ankle_x = FloatField(null=True)
    right_ankle_y = FloatField(null=True)
    right_ankle_z = FloatField(null=True)
    right_ankle_score = FloatField(null=True)
    right_ankle_is_visible = IntegerField(null=True)

    left_hip_x = FloatField(null=True)
    left_hip_y = FloatField(null=True)
    left_hip_z = FloatField(null=True)
    left_hip_score = FloatField(null=True)
    left_hip_is_visible = IntegerField(null=True)

    left_knee_x = FloatField(null=True)
    left_knee_y = FloatField(null=True)
    left_knee_z = FloatField(null=True)
    left_knee_score = FloatField(null=True)
    left_knee_is_visible = IntegerField(null=True)

    left_ankle_x = FloatField(null=True)
    left_ankle_y = FloatField(null=True)
    left_ankle_z = FloatField(null=True)
    left_ankle_score = FloatField(null=True)
    left_ankle_is_visible = IntegerField(null=True)

    right_eye_x = FloatField(null=True)
    right_eye_y = FloatField(null=True)
    right_eye_z = FloatField(null=True)
    right_eye_score = FloatField(null=True)
    right_eye_is_visible = IntegerField(null=True)

    left_eye_x = FloatField(null=True)
    left_eye_y = FloatField(null=True)
    left_eye_z = FloatField(null=True)
    left_eye_score = FloatField(null=True)
    left_eye_is_visible = IntegerField(null=True)

    right_ear_x = FloatField(null=True)
    right_ear_y = FloatField(null=True)
    right_ear_z = FloatField(null=True)
    right_ear_score = FloatField(null=True)
    right_ear_is_visible = IntegerField(null=True)

    left_ear_x = FloatField(null=True)
    left_ear_y = FloatField(null=True)
    left_ear_z = FloatField(null=True)
    left_ear_score = FloatField(null=True)
    left_ear_is_visible = IntegerField(null=True)

    hip_center_x = FloatField(null=True)
    hip_center_y = FloatField(null=True)
    hip_center_z = FloatField(null=True)
    hip_center_score = FloatField(null=True)
    hip_center_is_visible = IntegerField(null=True)

