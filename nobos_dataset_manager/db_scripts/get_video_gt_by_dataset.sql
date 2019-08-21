SELECT * FROM videogroundtruth, dataset
WHERE dataset.id = videogroundtruth.dataset_id AND
      dataset.name = 'SIM_2019_02_12_walks'