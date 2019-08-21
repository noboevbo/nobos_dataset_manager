from nobos_commons.data_structures.constants.dataset_part import DatasetPart

from nobos_dataset_manager.models.dataset import Dataset
from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.human import Human
from nobos_dataset_manager.models.human_action import HumanAction
from nobos_dataset_manager.models.video_ground_truth import VideoGroundTruth

train_data = (HumanAction.select(HumanAction)
              .join(VideoGroundTruth)
              .join(Dataset)
              .where(Dataset.name << ['SIM_2019_02_12_idle', 'SIM_2019_02_12_walks',
                                      'SIM_2019_02_12_walks_Near2cams', 'SIM_2019_02_12_waves'],
                     VideoGroundTruth.dataset_part == DatasetPart.TRAIN.value)
              .group_by(HumanAction.id)
              )

train_data_2 = (HumanAction.select(HumanAction)
                .join(VideoGroundTruth)
                .join(Dataset)
                .switch(HumanAction)
                .join(Human)
                .where(Dataset.name << ['SIM_2019_02_12_idle',
                                        'SIM_2019_02_12_walks',
                                        'SIM_2019_02_12_walks_Near2cams',
                                        'SIM_2019_02_12_waves'],
                       VideoGroundTruth.dataset_part == DatasetPart.TRAIN.value,
                       Human.datasource == Datasource.GROUND_TRUTH.value)
                .group_by(HumanAction.id)
                )
