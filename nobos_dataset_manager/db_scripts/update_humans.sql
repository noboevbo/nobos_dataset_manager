UPDATE human
SET datasource = 1
FROM framegroundtruth, videogroundtruth, dataset
WHERE human.frame_gt_id = framegroundtruth.id
AND framegroundtruth.video_gt_id = videogroundtruth.id
AND videogroundtruth.dataset_id = dataset.id
AND dataset.name != 'JHMDB'