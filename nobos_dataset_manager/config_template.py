from peewee import PostgresqlDatabase

from nobos_dataset_manager.configs.dataset_manager_config import DatasetManagerConfig

cfg = DatasetManagerConfig()
cfg.blob_storage_path = '/media/disks/gamma/nobos_dataset_manager'
cfg.db_conn = PostgresqlDatabase('ground_truth_store', host="localhost", port=1111, user="gt_worker", password="password")