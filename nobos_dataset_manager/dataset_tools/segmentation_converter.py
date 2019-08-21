import os

import cv2
import numpy as np
from nobos_commons.utils.file_helper import get_img_paths_from_folder, get_create_path, get_filename_from_path

rtSimColors = {
    "rest": np.array([0, 0, 0], dtype="uint8"),
    "guard_raily": np.array([0,0,255], dtype="uint8"),
    "car": np.array([0,255,255], dtype="uint8"),
    "wall": np.array([0,127,127], dtype="uint8"),
    "road": np.array([64, 64, 64], dtype="uint8"),
    "sidewalk": np.array([127, 127, 127], dtype="uint8"),
    "vegetatin": np.array([0, 255, 0], dtype="uint8"),
    "sky": np.array([127, 255, 255], dtype="uint8"),
    "pole_group": np.array([192, 192, 192], dtype="uint8"),
    "pedestrian": np.array([255, 127, 255], dtype="uint8"),
    "static": np.array([255, 255, 127], dtype="uint8"),
    "bridge": np.array([0, 0, 127], dtype="uint8"),
    "building": np.array([0, 127, 0], dtype="uint8"),
    "dynamic": np.array([255, 0, 127], dtype="uint8"),
    "traffic_light": np.array([127, 0, 127], dtype="uint8"),
    "traffic_sign": np.array([255, 0, 0], dtype="uint8"),
    "tunnel": np.array([0, 0, 160], dtype="uint8"),
    "fence": np.array([127, 0, 255], dtype="uint8"),
    "terrain": np.array([255, 255, 255], dtype="uint8"),
    "ground": np.array([127, 0, 0], dtype="uint8"),
}

cityscapes_colors = {
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    "unlabeled": np.array([0, 0, 0], dtype="uint8"),
    "ego vehicle": np.array([0, 0, 0], dtype="uint8"),
    "rectification border": np.array([0, 0, 0], dtype="uint8"),
    "out of roi": np.array([0, 0, 0], dtype="uint8"),
    "static": np.array([0, 0, 0], dtype="uint8"),
    "dynamic": np.array([111, 74, 0], dtype="uint8"),
    "ground": np.array([81, 0, 81], dtype="uint8"),
    "road": np.array([128, 64, 128], dtype="uint8"),
    "sidewalk": np.array([244, 35, 232], dtype="uint8"),
    "parking": np.array([250, 170, 160], dtype="uint8"),
    "rail track": np.array([230, 150, 140], dtype="uint8"),
    "building": np.array([70, 70, 70], dtype="uint8"),
    "wall": np.array([102, 102, 156], dtype="uint8"),
    "fence": np.array([190, 153, 153], dtype="uint8"),
    "guard rail": np.array([180, 165, 180], dtype="uint8"),
    "bridge": np.array([150, 100, 100], dtype="uint8"),
    "tunnel": np.array([150, 120, 90], dtype="uint8"),
    "pole": np.array([153, 153, 153], dtype="uint8"),
    "polegroup": np.array([153, 153, 153], dtype="uint8"),
    "traffic light": np.array([250, 170, 30], dtype="uint8"),
    "traffic sign": np.array([220, 220, 0], dtype="uint8"),
    "vegetation": np.array([107, 142, 35], dtype="uint8"),
    "terrain": np.array([152, 251, 152], dtype="uint8"),
    "sky": np.array([70, 130, 180], dtype="uint8"),
    "person": np.array([220, 20, 60], dtype="uint8"),
    "rider": np.array([255, 0, 0], dtype="uint8"),
    "car": np.array([0, 0, 142], dtype="uint8"),
    "truck": np.array([0, 0, 70], dtype="uint8"),
    "bus": np.array([0, 60, 100], dtype="uint8"),
    "caravan": np.array([0, 0, 90], dtype="uint8"),
    "trailer": np.array([0, 0, 110], dtype="uint8"),
    "train": np.array([0, 80, 100], dtype="uint8"),
    "motorcycle": np.array([0, 0, 230], dtype="uint8"),
    "bicycle": np.array([119, 11, 32], dtype="uint8"),
    "license plate": np.array([0, 0, 142], dtype="uint8"),
}

rtSimCityscapesMapping = {
    "rest": "unlabeled",
    "guard_raily": "guard rail",
    "car": "car",
    "wall": "wall",
    "road": "road",
    "sidewalk": "sidewalk",
    "vegetatin": "vegetation",
    "sky": "sky",
    "pole_group": "polegroup",
    "pedestrian": "person",
    "static": "static",
    "bridge": "bridge",
    "building": "building",
    "dynamic": "dynamic",
    "traffic_light": "traffic light",
    "traffic_sign": "traffic sign",
    "tunnel": "tunnel",
    "fence": "fence",
    "terrain": "terrain",
    "ground": "ground",
}

def get_by_value(dictionary, value):
    for name, val in dictionary.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if val == value:
            return name

if __name__ == "__main__":
    # RTSim to Cityscapes
    input_path = "/media/disks/gamma/records/simulation/Record-2019-08-19_14-08-26-1/Main Camera/"
    output_path = get_create_path(os.path.join(input_path, "cityscapes"))
    for img_path in get_img_paths_from_folder(input_path):
        filename = get_filename_from_path(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w, bpp = np.shape(img)
        for key, color in rtSimColors.items():
            img[np.where((img == color).all(axis=2))] = cityscapes_colors[rtSimCityscapesMapping[key]]
        # for py in range(0, h):
        #     for px in range(0, w):
        #         b = img[py][px][0]
        #         g = img[py][px][1]
        #         r = img[py][px][2]
        #         className = get_by_value(rtSimColors, (r, g, b))
        #         if className is None:
        #             a = 1
        #         cityscapesValues = cityscapes_colors[className]
        #         img[py][px][0] = cityscapesValues[2]
        #         img[py][px][1] = cityscapesValues[1]
        #         img[py][px][2] = cityscapesValues[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, filename), img)