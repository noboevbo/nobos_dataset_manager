from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman

from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth

skeleton = SkeletonStickman()

def remove_hardcoded_path(gt_row: VideoGroundTruth):
    vid_img_path = str(gt_row.vid_img_path)
    changed = False
    if vid_img_path.startswith("/media/disks/beta/nobos_dataset_manager/"):
        gt_row.vid_img_path = vid_img_path.replace("/media/disks/beta/nobos_dataset_manager/", "")
        changed = True
    vid_path = str(gt_row.vid_path)
    if vid_path.startswith("/media/disks/beta/nobos_dataset_manager/"):
        gt_row.vid_path = vid_path.replace("/media/disks/beta/nobos_dataset_manager/", "")
        changed = True
    if changed:
        print("Updated row!")
        gt_row.save()

def fix_data_manager_paths():
    train_data = (VideoGroundTruth.select(VideoGroundTruth))
    num_rows = train_data.count()
    count = 0
    for gt_row in train_data:
        print("Working on {}/{}".format(count, num_rows))
        remove_hardcoded_path(gt_row)
        count += 1

if __name__ == "__main__":
    fix_data_manager_paths()