class DatasetManagerConfig(object):
    __slots__ = ['blob_storage_path', 'db_conn']

    def __init__(self):
        self.blob_storage_path = None
        self.db_conn = None
