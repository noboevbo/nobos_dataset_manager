import os

import cv2
from typing import List

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.visualization.img_utils import add_img_title
from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton

from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.view_actiongroundtruth import ViewActionGroundTruth
from nobos_dataset_manager.utils import get_skeleton_from_skeleton_db


def get_from_db() -> List[ViewActionGroundTruth]:
    return (ViewActionGroundTruth.select(ViewActionGroundTruth)
            .where(ViewActionGroundTruth.dataset_name << ['SIM_2019_03_06-walk'],
                   ViewActionGroundTruth.vid_name == "Vue2",
                   ViewActionGroundTruth.dataset_part == DatasetPart.TRAIN.value,
                   ViewActionGroundTruth.human_datasource == Datasource.GROUND_TRUTH.value)
            .order_by(ViewActionGroundTruth.human_action_id, ViewActionGroundTruth.frame_num)
            )


if __name__ == "__main__":
    db_entries = get_from_db()
    for entry in db_entries:
        skeleton = get_skeleton_from_skeleton_db(entry)
        img = cv2.imread(os.path.join("/media/disks/beta/nobos_dataset_manager",
                                      entry.vid_img_path,
                                      "{}.jpg".format(str(entry.frame_num).zfill(6))))
        img = get_visualized_skeleton(img, skeleton)
        add_img_title(img, "{}: {}".format(str(entry.human_action_id), str(entry.frame_num)))
        cv2.imshow("preview", img)
        cv2.waitKey()
