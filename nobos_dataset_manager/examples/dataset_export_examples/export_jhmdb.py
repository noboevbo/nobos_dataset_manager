from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.humans_metadata.action import jhmdb_actions

from nobos_dataset_manager.dataset_exporters.ehpi_exporter import EhpiExporter
from nobos_dataset_manager.models.datasource import Datasource
from nobos_dataset_manager.models.view_actiongroundtruth import ViewActionGroundTruth


def gen_jhmdb(output_path: str, split: str, dataset_part: DatasetPart, datasource: Datasource):
    actions_to_load = (ViewActionGroundTruth.select(ViewActionGroundTruth.human_action_id)
                       .distinct()
                       .where(ViewActionGroundTruth.dataset_name == ['JHMDB'],
                              ViewActionGroundTruth.dataset_part == dataset_part.value,
                              ViewActionGroundTruth.dataset_split == split,
                              ViewActionGroundTruth.human_datasource == datasource.value)
                       )
    gt_per_action = []
    for action in actions_to_load:
        gt_per_action.append(ViewActionGroundTruth.select(ViewActionGroundTruth)
                             .where(ViewActionGroundTruth.human_action_id == action.human_action_id,
                                    ViewActionGroundTruth.dataset_name == ['JHMDB'],
                                    ViewActionGroundTruth.dataset_part == dataset_part.value,
                                    ViewActionGroundTruth.dataset_split == split,
                                    ViewActionGroundTruth.human_datasource == datasource.value)
                             .order_by(ViewActionGroundTruth.frame_num.asc()))
    exporter = EhpiExporter()
    exporter.get_ehpi_images(gt_per_action, actions_to_load.count(),
                             output_path=output_path,
                             dataset_part=dataset_part, actions=jhmdb_actions, to_many_zero_frames=30)


if __name__ == "__main__":
    # Export Train / Test of JHMDB-1
    gen_jhmdb("/media/disks/beta/datasets/ehpi/JHMDB_ITSC-1-GT", split='JHMDB-1', datasource=Datasource.GROUND_TRUTH,
              dataset_part=DatasetPart.TRAIN)
    gen_jhmdb("/media/disks/beta/datasets/ehpi/JHMDB_ITSC-1-GT", split='JHMDB-1', datasource=Datasource.GROUND_TRUTH,
              dataset_part=DatasetPart.TEST)