from datetime import date

from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman

from nobos_dataset_manager.models.base_model import db
from nobos_dataset_manager.models.bounding_box import BoundingBox
from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.frame_ground_truth import FrameGroundTruth
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.skeleton_joints import SkeletonJoints
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth

# if __name__ == "__main__":
#     db.connect()
#     db.drop_tables([BoundingBox, Dataset, Human, HumanAction, FrameGroundTruth, SkeletonJoints, VideoGroundTruth])
#     db.create_tables([BoundingBox, Dataset, Human, HumanAction, FrameGroundTruth, SkeletonJoints, VideoGroundTruth])
#
#     dataset = Dataset()
#     dataset.name = "JHMDB"
#     dataset.save()

    # vid_gt = VideoGroundTruth()
    # vid_gt.dataset = dataset
    #
    # vid_gt.vid_path = "/home/bla"
    # vid_gt.vid_name = "test"
    # vid_gt.num_frames = 20
    # vid_gt.viewport = "SSW"
    # vid_gt.created_date = date(2013, 1, 1)
    # vid_gt.save()
    #
    # frame_gt = FrameGroundTruth()
    # frame_gt.video_gt = vid_gt
    # frame_gt.frame_num = 0
    # frame_gt.__data__["frame_num"] = 20
    # frame_gt.save()
    #
    # skeleton = SkeletonStickman()
    #
    # skeleton_joints = SkeletonJoints()
    # x = skeleton_joints.__data__
    # for joint in skeleton.joints:
    #     skeleton_joints.__data__[joint.name + "_x"] = joint.x
    #     skeleton_joints.__data__[joint.name + "_y"] = joint.y
    #     skeleton_joints.__data__[joint.name + "_score"] = joint.score
    #     skeleton_joints.__data__[joint.name + "_is_visible"] = joint.visibility
    # skeleton_joints.save()
    #
    # human = Human()
    # human.uid = "abc"
    #
    # human.bounding_box = BoundingBox()
    # human.bounding_box.top_left_x = 2.0
    # human.bounding_box.top_left_y = 50.2
    # human.bounding_box.width = 200
    # human.bounding_box.height = 200
    # human.frame_gt = frame_gt
    # human.save()
    #
    # # query = (Human.select(Human, BoundingBox)
    # #          .join(BoundingBox, attr='bounding_box')
    # #          .order_by(Human.uid))
    # # for human in query:
    # #     print(human.bounding_box.width)
    # # human = Human.get(id=1)
    # # bb = human.bounding_box
    # # print(human.bounding_box.width)
    # # test = 1