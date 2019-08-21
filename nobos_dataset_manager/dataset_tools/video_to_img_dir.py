import os

from nobos_commons.utils.file_helper import get_video_paths_from_folder, get_create_path, \
    get_filename_without_extension, get_filename_from_path, get_video_paths_from_folder_recursive


def convert_all_vids_to_images(source_dir, fps: int, recursively: bool):
    if recursively:
        vid_paths = get_video_paths_from_folder_recursive(source_dir)
    else:
        vid_paths = get_video_paths_from_folder(source_dir)
    for filepath in vid_paths:
        raw_name = get_filename_without_extension(get_filename_from_path(filepath))
        output_dir = get_create_path(os.path.join(os.path.dirname(os.path.abspath(filepath)), raw_name))
        os.system("ffmpeg -i {0} -r {2} {1}/%6d.jpg".format(filepath, output_dir, fps))


if __name__ == "__main__":
    fps = 25
    # convert_all_vids_to_images("/media/disks/beta/records/real_cam/2019_03_13_Freilichtmuseum_Dashcam/Train/encoded/", fps)
    # convert_all_vids_to_images("/media/disks/beta/records/real_cam/2019_03_13_Freilichtmuseum_YI/Train/encoded/",
    #                            fps)
    # convert_all_vids_to_images("/media/disks/beta/records/real_cam/2019_03_13_Freilichtmuseum_YI/Test/encoded/",
    #                            fps)
    convert_all_vids_to_images("/home/dennis/Downloads/hmdb51_org/", fps=fps, recursively=True)