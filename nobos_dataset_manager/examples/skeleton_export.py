import os

import cv2
from typing import List

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.humans_metadata.action import Action
from nobos_commons.utils.file_helper import get_create_path
from nobos_commons.visualization.img_utils import add_img_title
from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton

from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.view_actiongroundtruth import ViewActionGroundTruth
from nobos_dataset_manager.utils import get_skeleton_from_skeleton_db


def get_from_db(seq_id) -> List[ViewActionGroundTruth]:
    return (ViewActionGroundTruth.select(ViewActionGroundTruth)
            .where(
                   ViewActionGroundTruth.human_action_id == seq_id)
            .order_by(ViewActionGroundTruth.human_action_id, ViewActionGroundTruth.frame_num)
            )

def export_seqs_its():
    # seq_ids = [1940, 1956, 1975, 1996, 2014]
    seq_ids = [ 3836]
    for seq_id in seq_ids:
        db_entries = get_from_db(seq_id)
        out_path = get_create_path("/media/disks/beta/dump/false_detected_sequences/pose/{}".format(seq_id))
        frame_count = 0
        for frame_num, entry in enumerate(db_entries):
            # if frame_num % 2 != 0:
            #     continue
            skeleton = get_skeleton_from_skeleton_db(entry)
            img = cv2.imread(os.path.join("/media/nobos_datastore/",
                                          entry.vid_img_path,
                                          "{}.png".format(str(entry.frame_num).zfill(6))))
            img = get_visualized_skeleton(img, skeleton)
            add_img_title(img, "{}: {}".format(str(entry.human_action_id), str(entry.frame_num)))
            # cv2.imshow("preview", img)
            print(frame_num)

            cv2.imwrite(os.path.join(out_path, "{}.jpg".format(str.zfill(str(frame_count), 5))), img)
            frame_count += 1
            # cv2.waitKey()

def export_sim_its():
    db_entries = (ViewActionGroundTruth.select(ViewActionGroundTruth)
            .where(ViewActionGroundTruth.dataset_name << ['SIM_2019_03_06-sit'],
                   ViewActionGroundTruth.vid_name == "Vue1",
                   ViewActionGroundTruth.frame_num == 1533)
            .order_by(ViewActionGroundTruth.human_action_id, ViewActionGroundTruth.frame_num)
            )
    frame_count = 0
    for frame_num, entry in enumerate(db_entries):
        skeleton = get_skeleton_from_skeleton_db(entry)
        img = cv2.imread(os.path.join("/media/nobos_datastore/",
                                      entry.vid_img_path,
                                      "{}.jpg".format(str(entry.frame_num).zfill(6))))
        img = get_visualized_skeleton(img, skeleton)
        add_img_title(img, "{}: {}".format(str(entry.human_action_id), str(entry.frame_num)))
        cv2.imshow("preview", img)
        print(frame_num)

        # cv2.imwrite(os.path.join(out_path, "{}.jpg".format(str.zfill(str(frame_count), 5))), img)
        frame_count += 1
        cv2.waitKey()

if __name__ == "__main__":
    export_seqs_its()
    # export_sim_its()