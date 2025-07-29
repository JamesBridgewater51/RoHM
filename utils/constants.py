SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
# SMPL joint sets for masking schemes
HEAD_ID_SET = {15}  # head
HANDS_IDS_SET = {20, 21}  # SMPL wrist joints (L_Wrist, R_Wrist) - often referred to as hands
FEET_IDS_SET = {10, 11}  # SMPL foot joints (L_Foot, R_Foot)

NECK_ID_SET = {12}
SPINE_IDS_SET = {3, 6, 9}  # Spine1, Spine2, Spine3
PELVIS_ID_SET = {0} # Often considered the root
COLLAR_IDS_SET = {13, 14}  # L_Collar, R_Collar
SHOULDER_IDS_SET = {16, 17}  # L_Shoulder, R_Shoulder
ELBOW_IDS_SET = {18, 19}  # L_Elbow, R_Elbow
HIP_IDS_SET = {1, 2}  # L_Hip, R_Hip
KNEE_IDS_SET = {4, 5}  # L_Knee, R_Knee
ANKLE_IDS_SET = {7, 8}  # L_Ankle, R_Ankle

# Individual joint IDs based on SMPL_JOINT_NAMES order
# Pelvis (root) is 0
LEFT_HIP_ID = 1
RIGHT_HIP_ID = 2
LEFT_KNEE_ID = 4
RIGHT_KNEE_ID = 5
LEFT_ANKLE_ID = 7
RIGHT_ANKLE_ID = 8
LEFT_FOOT_ID = 10
RIGHT_FOOT_ID = 11
# Neck is 12
# Head is 15
LEFT_SHOULDER_ID = 16
RIGHT_SHOULDER_ID = 17
LEFT_ELBOW_ID = 18
RIGHT_ELBOW_ID = 19
LEFT_WRIST_ID = 20 # Referred to as left hand
RIGHT_WRIST_ID = 21 # Referred to as right hand


# Individual joint ID sets
LEFT_HIP_ID_SET = {LEFT_HIP_ID}
RIGHT_HIP_ID_SET = {RIGHT_HIP_ID}
LEFT_FOOT_ID_SET = {LEFT_FOOT_ID}
RIGHT_FOOT_ID_SET = {RIGHT_FOOT_ID}
LEFT_SHOULDER_ID_SET = {LEFT_SHOULDER_ID}
RIGHT_SHOULDER_ID_SET = {RIGHT_SHOULDER_ID}
LEFT_HAND_ID_SET = {LEFT_WRIST_ID} # Using wrist as hand
RIGHT_HAND_ID_SET = {RIGHT_WRIST_ID} # Using wrist as hand
LEFT_KNEE_ID_SET = {LEFT_KNEE_ID}
RIGHT_KNEE_ID_SET = {RIGHT_KNEE_ID}
LEFT_ANKLE_ID_SET = {LEFT_ANKLE_ID}
RIGHT_ANKLE_ID_SET = {RIGHT_ANKLE_ID}
LEFT_ELBOW_ID_SET = {LEFT_ELBOW_ID}
RIGHT_ELBOW_ID_SET = {RIGHT_ELBOW_ID}


# Assuming 22 SMPL joints (0-21) based on SMPL_JOINT_NAMES
ALL_SMPL_JOINT_IDS_SET = set(range(len(SMPL_JOINT_NAMES)))


MOCAP_MASK_SCHEME_CHOICES = [
    "head_only",
    "head_with_two_hands",
    "head_with_two_hands_and_two_feets",
    "upper_body",
    "lower_body",
    "core_body",
    "extremities_only",
    "full_body_keep_all",
    "torso_and_head",
    # New schemes
    "both_hips_only",
    "hips_with_root",
    "both_feet_only",
    "both_shoulders_only",
    "both_hands_only", # Both wrists
    "both_knees_only",
    "both_ankles_only",
    "both_elbows_only",
    "pelvis_only", # Root only
]

MOCAP_VIS_MASK_IDS = {
    "head_only": HEAD_ID_SET,
    "head_with_two_hands": HEAD_ID_SET.union(HANDS_IDS_SET),
    "head_with_two_hands_and_two_feets": HEAD_ID_SET.union(HANDS_IDS_SET).union(FEET_IDS_SET),
    "upper_body": HEAD_ID_SET.union(NECK_ID_SET).union(SPINE_IDS_SET).union(COLLAR_IDS_SET).union(SHOULDER_IDS_SET).union(ELBOW_IDS_SET).union(HANDS_IDS_SET),
    "lower_body": PELVIS_ID_SET.union(HIP_IDS_SET).union(KNEE_IDS_SET).union(ANKLE_IDS_SET).union(FEET_IDS_SET),
    "core_body": PELVIS_ID_SET.union(SPINE_IDS_SET).union(NECK_ID_SET).union(HEAD_ID_SET),
    "extremities_only": HANDS_IDS_SET.union(FEET_IDS_SET),
    "full_body_keep_all": ALL_SMPL_JOINT_IDS_SET,
    "torso_and_head": PELVIS_ID_SET.union(SPINE_IDS_SET).union(NECK_ID_SET).union(HEAD_ID_SET).union(COLLAR_IDS_SET).union(SHOULDER_IDS_SET),
    # New scheme definitions
    "both_hips_only": HIP_IDS_SET,
    "hips_with_root": PELVIS_ID_SET.union(HIP_IDS_SET),
    "both_feet_only": FEET_IDS_SET,
    "both_shoulders_only": SHOULDER_IDS_SET,
    "both_hands_only": HANDS_IDS_SET,
    "both_knees_only": KNEE_IDS_SET,
    "both_ankles_only": ANKLE_IDS_SET,
    "both_elbows_only": ELBOW_IDS_SET,
    "pelvis_only": PELVIS_ID_SET,
}