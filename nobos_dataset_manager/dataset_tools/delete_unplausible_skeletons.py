import os

import cv2

from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton

from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.view_actiongroundtruth import ViewActionGroundTruth
from nobos_dataset_manager.utils import get_skeleton_from_skeleton_db

skeleton = SkeletonStickman()

def is_skeleton_plausible(gt_row: ViewActionGroundTruth, debug: bool = False):
    max_width = gt_row.frame_width
    max_height = gt_row.frame_height
    plausible_joints = 0
    for joint in skeleton.joints:
        x = gt_row.__data__[joint.name + "_x"]
        y = gt_row.__data__[joint.name + "_y"]
        if x is None or y is None or x < 0 or x > max_width or y < 0 or y > max_height:
            continue
        plausible_joints += 1
    if plausible_joints > 5: # Minimum 5 joints should be visible
        return True
    if debug:
        skeleton_view = get_skeleton_from_skeleton_db(gt_row)
        img_path = os.path.join("/media/disks/beta/nobos_dataset_manager/", gt_row.vid_img_path, "{}.png".format(str(gt_row.frame_num).zfill(6)))
        img = cv2.imread(img_path)
        img = get_visualized_skeleton(img, skeleton_view)
        cv2.imshow("test", img)
        cv2.waitKey(0)
    return False
#TODO: Bei JHMDB mal anzeigen lassen was das soll..


def delete_unplausible_skeletons():
    train_data = (ViewActionGroundTruth.select(ViewActionGroundTruth).where(ViewActionGroundTruth.dataset_name != "JHMDB"))
    num_rows = train_data.count()
    count = 0
    deleted_count = 0
    for gt_row in train_data:
        gt_row: ViewActionGroundTruth = gt_row # Just for intellisense
        if count % 10000 == 0:
            print("Working on {}/{}".format(count, num_rows))
        plausible = is_skeleton_plausible(gt_row, debug=False)
        if not plausible:
            status = Human.delete_by_id(gt_row.human_id)
            print("Deleted: Return status [Source: {}]: {}".format(Datasource(gt_row.human_datasource).name, status))
            deleted_count += 1
        count += 1

    print("Deleted {} humans".format(deleted_count))

if __name__ == "__main__":
    delete_unplausible_skeletons()