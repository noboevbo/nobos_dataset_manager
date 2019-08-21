from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.skeletons.skeleton_jhmdb import SkeletonJhmdb
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.tools.skeleton_converters.skeleton_converter_factory import SkeletonConverterFactory

from nobos_dataset_manager import configurator
from nobos_dataset_manager.config import cfg
from nobos_dataset_manager.dataset_importers.jhmdb_importer import JhmdbImporter, set_splits_for_video_gt
from nobos_dataset_manager.dataset_tools.pipeline_ehpi import PipelineEhpi
from nobos_dataset_manager.models.datasource import Datasource

if __name__ == "__main__":
    configurator.setup()
    skeleton_converter_factory = SkeletonConverterFactory()

    ## STEP 1: Import JHMDB Ground Truth
    importer = JhmdbImporter("/home/dennis/Downloads/JHMDB",
                             skeleton_converter_factory.get_skeleton_converter(SkeletonJhmdb, SkeletonStickman),
                             is_additional_source=False,
                             datasource=Datasource.GROUND_TRUTH,
                             img_dir_to_skeleton_tool=PipelineEhpi(use_pose_tracking=True, image_size=ImageSize(320, 240),
                                 use_fast_mode=False))
    # importer.import_data()
    with cfg.db_conn.atomic() as transaction:  # Opens new transaction.
        try:
            importer.import_data()
        except Exception as err:
            # Because this block of code is wrapped with "atomic", a
            # new transaction will begin automatically after the call
            # to rollback().
            print(err)
            transaction.rollback()

    ## STEP 2: Import JHMDB Pose Algo Results
    importer = JhmdbImporter("/home/dennis/Downloads/JHMDB",
                             skeleton_converter_factory.get_skeleton_converter(SkeletonJhmdb, SkeletonStickman),
                             is_additional_source=True,
                             datasource=Datasource.OFP_POSE_RECOGNITION,
                             img_dir_to_skeleton_tool=PipelineEhpi(use_pose_tracking=True, image_size=ImageSize(320, 240),
                                 use_fast_mode=False))
    # importer.import_data()
    with cfg.db_conn.atomic() as transaction:  # Opens new transaction.
        try:
            importer.import_data()
        except Exception as err:
            # Because this block of code is wrapped with "atomic", a
            # new transaction will begin automatically after the call
            # to rollback().
            print(err)
            transaction.rollback()

    # Step 3: Setup Dataset splits
    with cfg.db_conn.atomic() as transaction:  # Opens new transaction.
        try:
            set_splits_for_video_gt("/home/dennis/Downloads/JHMDB/splits", 1, is_subsplit=False)
            set_splits_for_video_gt("/home/dennis/Downloads/JHMDB/splits", 2, is_subsplit=False)
            set_splits_for_video_gt("/home/dennis/Downloads/JHMDB/splits", 3, is_subsplit=False)
        except Exception as err:
            # Because this block of code is wrapped with "atomic", a
            # new transaction will begin automatically after the call
            # to rollback().
            print(err)
            transaction.rollback()

