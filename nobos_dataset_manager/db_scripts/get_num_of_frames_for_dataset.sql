SELECT COUNT(DISTINCT framegroundtruth.id), dataset.name
FROM videogroundtruth,
     framegroundtruth,
     dataset
WHERE framegroundtruth.video_gt_id = videogroundtruth.id
  AND dataset.id = videogroundtruth.dataset_id
GROUP BY dataset.name
