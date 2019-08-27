CREATE VIEW view_actiongroundtruth AS
  SELECT dataset.id as dataset_id,
         dataset.name as dataset_name,

         videogroundtruth.id as video_gt_id,
         videogroundtruth.vid_path,
         videogroundtruth.vid_img_path,
         videogroundtruth.vid_segmentation_path ,
         videogroundtruth.vid_depth_path,
         videogroundtruth.vid_name,
         videogroundtruth.fps,
         datasetsplit.dataset_part,
         datasetsplit.name as dataset_split,
         videogroundtruth.num_frames,
         videogroundtruth.frame_width,
         videogroundtruth.frame_height,
         videogroundtruth.viewport,
         videogroundtruth.created_date,

         framegroundtruth.id as frame_gt_id,
         framegroundtruth.frame_num,

         humanaction.id as human_action_id,
         humanaction.action,

         human.id AS human_id,
         human.uid AS human_uid,
         human.datasource as human_datasource,

         boundingbox.id AS bb_id,
         boundingbox.top_left_x AS bb_top_left_x,
         boundingbox.top_left_y AS bb_top_left_y,
         boundingbox.top_left_z AS bb_top_left_z,
         boundingbox.width AS bb_width,
         boundingbox.height AS bb_height,
         boundingbox.depth AS bb_depth,

         skeletonjoints.id as skeleton_joints_id,
         skeletonjoints.nose_x,
         skeletonjoints.nose_y,
         skeletonjoints.nose_z,
         skeletonjoints.nose_score,
         skeletonjoints.nose_is_visible,

         skeletonjoints.neck_x,
         skeletonjoints.neck_y,
         skeletonjoints.neck_z,
         skeletonjoints.neck_score,
         skeletonjoints.neck_is_visible,

         skeletonjoints.right_shoulder_x,
         skeletonjoints.right_shoulder_y,
         skeletonjoints.right_shoulder_z,
         skeletonjoints.right_shoulder_score,
         skeletonjoints.right_shoulder_is_visible,

         skeletonjoints.right_elbow_x,
         skeletonjoints.right_elbow_y,
         skeletonjoints.right_elbow_z,
         skeletonjoints.right_elbow_score,
         skeletonjoints.right_elbow_is_visible,

         skeletonjoints.right_wrist_x,
         skeletonjoints.right_wrist_y,
         skeletonjoints.right_wrist_z,
         skeletonjoints.right_wrist_score,
         skeletonjoints.right_wrist_is_visible,

         skeletonjoints.left_shoulder_x,
         skeletonjoints.left_shoulder_y,
         skeletonjoints.left_shoulder_z,
         skeletonjoints.left_shoulder_score,
         skeletonjoints.left_shoulder_is_visible,

         skeletonjoints.left_elbow_x,
         skeletonjoints.left_elbow_y,
         skeletonjoints.left_elbow_z,
         skeletonjoints.left_elbow_score,
         skeletonjoints.left_elbow_is_visible,

         skeletonjoints.left_wrist_x,
         skeletonjoints.left_wrist_y,
         skeletonjoints.left_wrist_z,
         skeletonjoints.left_wrist_score,
         skeletonjoints.left_wrist_is_visible,

         skeletonjoints.right_hip_x,
         skeletonjoints.right_hip_y,
         skeletonjoints.right_hip_z,
         skeletonjoints.right_hip_score,
         skeletonjoints.right_hip_is_visible,

         skeletonjoints.right_knee_x,
         skeletonjoints.right_knee_y,
         skeletonjoints.right_knee_z,
         skeletonjoints.right_knee_score,
         skeletonjoints.right_knee_is_visible,

         skeletonjoints.right_ankle_x,
         skeletonjoints.right_ankle_y,
         skeletonjoints.right_ankle_z,
         skeletonjoints.right_ankle_score,
         skeletonjoints.right_ankle_is_visible,

         skeletonjoints.left_hip_x,
         skeletonjoints.left_hip_y,
         skeletonjoints.left_hip_z,
         skeletonjoints.left_hip_score,
         skeletonjoints.left_hip_is_visible,

         skeletonjoints.left_knee_x,
         skeletonjoints.left_knee_y,
         skeletonjoints.left_knee_z,
         skeletonjoints.left_knee_score,
         skeletonjoints.left_knee_is_visible,

         skeletonjoints.left_ankle_x,
         skeletonjoints.left_ankle_y,
         skeletonjoints.left_ankle_z,
         skeletonjoints.left_ankle_score,
         skeletonjoints.left_ankle_is_visible,

         skeletonjoints.right_eye_x,
         skeletonjoints.right_eye_y,
         skeletonjoints.right_eye_z,
         skeletonjoints.right_eye_score,
         skeletonjoints.right_eye_is_visible,

         skeletonjoints.left_eye_x,
         skeletonjoints.left_eye_y,
         skeletonjoints.left_eye_z,
         skeletonjoints.left_eye_score,
         skeletonjoints.left_eye_is_visible,

         skeletonjoints.right_ear_x,
         skeletonjoints.right_ear_y,
         skeletonjoints.right_ear_z,
         skeletonjoints.right_ear_score,
         skeletonjoints.right_ear_is_visible,

         skeletonjoints.left_ear_x,
         skeletonjoints.left_ear_y,
         skeletonjoints.left_ear_z,
         skeletonjoints.left_ear_score,
         skeletonjoints.left_ear_is_visible,

         skeletonjoints.hip_center_x,
         skeletonjoints.hip_center_y,
         skeletonjoints.hip_center_z,
         skeletonjoints.hip_center_score,
         skeletonjoints.hip_center_is_visible
FROM skeletonjoints, human, humanaction, boundingbox, videogroundtruth, datasetsplit,
     datasetsplit_videogroundtruth_through, framegroundtruth, dataset
WHERE
skeletonjoints.human_id = human.id
AND boundingbox.human_id = human.id
AND human.action_id = humanaction.id
AND human.frame_gt_id = framegroundtruth.id
AND datasetsplit_videogroundtruth_through.datasetsplit_id = datasetsplit.id
AND datasetsplit_videogroundtruth_through.videogroundtruth_id = videogroundtruth.id
AND framegroundtruth.video_gt_id = videogroundtruth.id
AND videogroundtruth.dataset_id = dataset.id;

alter view view_actiongroundtruth owner to gt_worker; -- change username if other than gt_worker.