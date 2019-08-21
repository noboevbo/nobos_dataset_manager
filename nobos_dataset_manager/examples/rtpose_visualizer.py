import cv2
import nobos_dataset_manager.config

from nobos_commons.utils.bounding_box_helper import get_human_bounding_box_from_joints
from nobos_commons.visualization.detection_visualizer import draw_bb
from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton
from nobos_dataset_api.rtsim.rtsim_pose import read_annotation_2d, get_skeleton_from_row


def visualize(gt_file_path: str):
    gt_df = read_annotation_2d(gt_file_path)
    for index, row in gt_df.iterrows():
        frame_num = row["frame_num"]
        if frame_num != 34:
            continue
        img_path = row["img_path"]
        img = cv2.imread(img_path)


        human_uid = row['uid']
        action_name = row['action']

        skeleton = get_skeleton_from_row(row)
        bb = get_human_bounding_box_from_joints(skeleton.joints)

        get_visualized_skeleton(img, skeleton)
        draw_bb(img, bb, "Sit", thickness=4)
        cv2.imwrite("/media/disks/beta/dump/new_sim_examples/sit_bb.png", img)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)

if __name__ == "__main__":
    visualize("/media/disks/beta/dump/new_sim_examples/Record-2019-04-18_14-42-31-3/annotations/pose-view_Vue1.txt")