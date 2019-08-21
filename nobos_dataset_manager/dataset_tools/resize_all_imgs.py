import os

import cv2
from nobos_commons.data_structures.dimension import ImageSize

from nobos_commons.utils.file_helper import get_immediate_subdirectories, get_last_dir_name, get_img_paths_from_folder, \
    get_filename_from_path, get_create_path

if __name__ == "__main__":
    target_image_size = ImageSize(640, 480)
    src = "/media/disks/beta/tmp_new_import/2019_02_27_Hella_OFP_ParkingLot/imgs/"
    output_folder = "/media/disks/beta/tmp_rescaled/2019_02_27_Hella_OFP_ParkingLot/"
    for sub_folder in get_immediate_subdirectories(src):
        dir_name = get_last_dir_name(sub_folder)
        output_path = get_create_path(os.path.join(output_folder, dir_name))
        for img_path in get_img_paths_from_folder(sub_folder):
            filename = get_filename_from_path(img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (target_image_size.width, target_image_size.height))
            cv2.imwrite(os.path.join(output_path, filename), img)