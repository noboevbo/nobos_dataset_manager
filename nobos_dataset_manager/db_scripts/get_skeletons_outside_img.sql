SELECT videogroundtruth.id, framegroundtruth.id FROM skeletonjoints, human, framegroundtruth, videogroundtruth
WHERE
(skeletonjoints.human_id = human.id AND
    human.frame_gt_id = framegroundtruth.id AND
    framegroundtruth.video_gt_id = videogroundtruth.id)
AND
  (
    ( nose_x < 0 or nose_y < 0 or nose_x > videogroundtruth.frame_width or nose_y > videogroundtruth.frame_height)
    and ( neck_x < 0 or neck_y < 0 or neck_x > videogroundtruth.frame_width or neck_y > videogroundtruth.frame_height)
    and ( right_shoulder_x < 0 or right_shoulder_y < 0 or right_shoulder_x > videogroundtruth.frame_width or right_shoulder_y > videogroundtruth.frame_height)
    and ( right_elbow_x < 0 or right_elbow_y < 0 or  right_elbow_x > videogroundtruth.frame_width or right_elbow_y > videogroundtruth.frame_height)
    and ( right_wrist_x < 0 or right_wrist_y < 0 or right_wrist_x > videogroundtruth.frame_width or right_wrist_y > videogroundtruth.frame_height)
    and ( left_shoulder_x < 0 or left_shoulder_y < 0 or left_shoulder_x > videogroundtruth.frame_width or left_shoulder_y > videogroundtruth.frame_height)
    and ( left_elbow_x < 0 or left_elbow_y < 0 or left_elbow_x > videogroundtruth.frame_width or left_elbow_y > videogroundtruth.frame_height)
    and ( left_wrist_x < 0 or left_wrist_y < 0 or left_wrist_x > videogroundtruth.frame_width or left_wrist_y > videogroundtruth.frame_height)
    and ( right_hip_x < 0 or right_hip_y < 0 or right_hip_x > videogroundtruth.frame_width or right_hip_y > videogroundtruth.frame_height)
    and ( right_knee_x < 0 or right_knee_y < 0 or right_knee_x > videogroundtruth.frame_width or right_knee_y > videogroundtruth.frame_height)
    and ( right_ankle_x < 0 or right_ankle_y < 0 or right_ankle_x > videogroundtruth.frame_width or right_ankle_y > videogroundtruth.frame_height)
    and ( left_hip_x < 0 or left_hip_y < 0 or left_hip_x > videogroundtruth.frame_width or left_hip_y > videogroundtruth.frame_height)
    and ( left_knee_x < 0 or left_knee_y < 0 or left_knee_x > videogroundtruth.frame_width or left_knee_y > videogroundtruth.frame_height)
    and ( left_ankle_x < 0 or left_ankle_y < 0 or left_ankle_x > videogroundtruth.frame_width or left_ankle_y > videogroundtruth.frame_height)
    and ( right_eye_x < 0 or right_eye_y < 0 or right_eye_x > videogroundtruth.frame_width or right_eye_y > videogroundtruth.frame_height)
    and ( left_eye_x < 0 or left_eye_y < 0 or left_eye_x > videogroundtruth.frame_width or left_eye_y > videogroundtruth.frame_height)
    and ( right_ear_x < 0 or right_ear_y < 0 or right_ear_x > videogroundtruth.frame_width or right_ear_y > videogroundtruth.frame_height)
    and ( left_ear_x < 0 or left_ear_y < 0 or left_ear_x > videogroundtruth.frame_width or left_ear_y > videogroundtruth.frame_height)
    and ( hip_center_x < 0 or hip_center_y < 0 or hip_center_x > videogroundtruth.frame_width or hip_center_y > videogroundtruth.frame_height)
  )