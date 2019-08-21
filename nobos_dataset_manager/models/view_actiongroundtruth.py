from peewee import FloatField, IntegerField, CharField, DateTimeField

from nobos_dataset_manager.models.base_model import BaseViewModel


class ViewActionGroundTruth(BaseViewModel):
    dataset_id = IntegerField()
    dataset_name = CharField()

    video_gt_id = IntegerField()
    vid_path = CharField(unique=True, null=True)
    vid_img_path = CharField(unique=True, null=True)
    vid_name = CharField()
    fps = IntegerField()
    dataset_part = IntegerField()
    dataset_split = CharField()
    num_frames = IntegerField()
    frame_width = IntegerField()
    frame_height = IntegerField()
    viewport = CharField(null=True)
    created_date = DateTimeField(null=True)

    frame_gt_id = IntegerField()
    frame_num = IntegerField()

    human_action_id = IntegerField()
    action = IntegerField()

    human_id = IntegerField()
    human_uid = CharField()
    human_datasource = IntegerField()

    bb_id = IntegerField
    bb_top_left_x = FloatField()
    bb_top_left_y = FloatField()
    bb_width = FloatField()
    bb_height = FloatField()

    skeleton_joints_id = IntegerField()
    nose_x = FloatField(null=True)
    nose_y = FloatField(null=True)
    nose_score = FloatField(null=True)
    nose_is_visible = IntegerField(null=True)
    neck_x = FloatField(null=True)
    neck_y = FloatField(null=True)
    neck_score = FloatField(null=True)
    neck_is_visible = IntegerField(null=True)
    right_shoulder_x = FloatField(null=True)
    right_shoulder_y = FloatField(null=True)
    right_shoulder_score = FloatField(null=True)
    right_shoulder_is_visible = IntegerField(null=True)
    right_elbow_x = FloatField(null=True)
    right_elbow_y = FloatField(null=True)
    right_elbow_score = FloatField(null=True)
    right_elbow_is_visible = IntegerField(null=True)
    right_wrist_x = FloatField(null=True)
    right_wrist_y = FloatField(null=True)
    right_wrist_score = FloatField(null=True)
    right_wrist_is_visible = IntegerField(null=True)
    left_shoulder_x = FloatField(null=True)
    left_shoulder_y = FloatField(null=True)
    left_shoulder_score = FloatField(null=True)
    left_shoulder_is_visible = IntegerField(null=True)
    left_elbow_x = FloatField(null=True)
    left_elbow_y = FloatField(null=True)
    left_elbow_score = FloatField(null=True)
    left_elbow_is_visible = IntegerField(null=True)
    left_wrist_x = FloatField(null=True)
    left_wrist_y = FloatField(null=True)
    left_wrist_score = FloatField(null=True)
    left_wrist_is_visible = IntegerField(null=True)
    right_hip_x = FloatField(null=True)
    right_hip_y = FloatField(null=True)
    right_hip_score = FloatField(null=True)
    right_hip_is_visible = IntegerField(null=True)
    right_knee_x = FloatField(null=True)
    right_knee_y = FloatField(null=True)
    right_knee_score = FloatField(null=True)
    right_knee_is_visible = IntegerField(null=True)
    right_ankle_x = FloatField(null=True)
    right_ankle_y = FloatField(null=True)
    right_ankle_score = FloatField(null=True)
    right_ankle_is_visible = IntegerField(null=True)
    left_hip_x = FloatField(null=True)
    left_hip_y = FloatField(null=True)
    left_hip_score = FloatField(null=True)
    left_hip_is_visible = IntegerField(null=True)
    left_knee_x = FloatField(null=True)
    left_knee_y = FloatField(null=True)
    left_knee_score = FloatField(null=True)
    left_knee_is_visible = IntegerField(null=True)
    left_ankle_x = FloatField(null=True)
    left_ankle_y = FloatField(null=True)
    left_ankle_score = FloatField(null=True)
    left_ankle_is_visible = IntegerField(null=True)
    right_eye_x = FloatField(null=True)
    right_eye_y = FloatField(null=True)
    right_eye_score = FloatField(null=True)
    right_eye_is_visible = IntegerField(null=True)
    left_eye_x = FloatField(null=True)
    left_eye_y = FloatField(null=True)
    left_eye_score = FloatField(null=True)
    left_eye_is_visible = IntegerField(null=True)
    right_ear_x = FloatField(null=True)
    right_ear_y = FloatField(null=True)
    right_ear_score = FloatField(null=True)
    right_ear_is_visible = IntegerField(null=True)
    left_ear_x = FloatField(null=True)
    left_ear_y = FloatField(null=True)
    left_ear_score = FloatField(null=True)
    left_ear_is_visible = IntegerField(null=True)
    hip_center_x = FloatField(null=True)
    hip_center_y = FloatField(null=True)
    hip_center_score = FloatField(null=True)
    hip_center_is_visible = IntegerField(null=True)

    class Meta:
        db_table = "view_actiongroundtruth"
