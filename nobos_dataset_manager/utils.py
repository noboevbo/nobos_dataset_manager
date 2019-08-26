from nobos_commons.data_structures.bounding_box_3D import BoundingBox3D
from nobos_commons.data_structures.skeletons.joint_3d import Joint3D
from nobos_commons.data_structures.skeletons.joint_visibility import JointVisibility
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.data_structures.skeletons.skeleton_stickman_3d import SkeletonStickman3D
from nobos_commons.utils.bounding_box_helper import get_human_bounding_box_from_joints, \
    get_human_bounding_box_3D_from_joints

from nobos_dataset_manager.models import bounding_box
from nobos_dataset_manager.models.bounding_box import BoundingBox
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.skeleton_joints import SkeletonJoints


def get_skeleton_db_from_skeleton(skeleton: SkeletonStickman3D, human: Human) -> SkeletonJoints:
    skeleton_joints = SkeletonJoints()
    skeleton_joints.human = human
    for joint in skeleton.joints:
        if not joint.is_set or joint.score <= 0 or (joint.x == 0 and joint.y == 0):
            continue
        skeleton_joints.__data__[joint.name + "_x"] = joint.x
        skeleton_joints.__data__[joint.name + "_y"] = joint.y
        if type(joint) is Joint3D:
            skeleton_joints.__data__[joint.name + "_z"] = joint.z
        skeleton_joints.__data__[joint.name + "_score"] = joint.score
        skeleton_joints.__data__[joint.name + "_is_visible"] = joint.visibility.value
    return skeleton_joints


def get_skeleton_from_skeleton_db(skeleton_joints: SkeletonJoints):
    skeleton = SkeletonStickman()
    for joint in skeleton.joints:
        x = skeleton_joints.__data__[joint.name + "_x"]
        if x is None:
            continue
        joint.x = x
        joint.y = skeleton_joints.__data__[joint.name + "_y"]
        joint.score = skeleton_joints.__data__[joint.name + "_score"]
        joint.visibility = JointVisibility(skeleton_joints.__data__[joint.name + "_is_visible"])
    return skeleton


def get_bb_db_from_skeleton(skeleton: SkeletonStickman, human: Human):
    bb_calculated: bounding_box.BoundingBox = get_human_bounding_box_from_joints(skeleton.joints)
    bb = BoundingBox()
    bb.human = human
    bb.top_left_x = bb_calculated.top_left.x
    bb.top_left_y = bb_calculated.top_left.y
    bb.width = bb_calculated.width
    bb.height = bb_calculated.height
    return bb

def get_bb_3D_db_from_skeleton(skeleton: SkeletonStickman, human: Human):
    bb_calculated: bounding_box.BoundingBox = get_human_bounding_box_3D_from_joints(skeleton.joints)
    bb = BoundingBox()
    bb.human = human
    bb.top_left_x = bb_calculated.top_left.x
    bb.top_left_y = bb_calculated.top_left.y
    bb.top_left_z = bb_calculated.top_left.z
    bb.width = bb_calculated.width
    bb.height = bb_calculated.height
    bb.depth = bb_calculated.depth
    return bb
