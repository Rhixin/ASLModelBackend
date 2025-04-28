import numpy as np

def extract_features_from_landmarks(landmarks_flat):
    landmarks = np.array(landmarks_flat).reshape(21, 3)
    
    wrist = landmarks[0]
    palm_indices = [0, 5, 9, 13, 17]
    palm_center = np.mean(landmarks[palm_indices], axis=0)
    
    features_dict = {}
    
    fingers = [
        [(1, 2, 3), (2, 3, 4)],    # Thumb
        [(5, 6, 7), (6, 7, 8)],    # Index
        [(9, 10, 11), (10, 11, 12)], # Middle
        [(13, 14, 15), (14, 15, 16)], # Ring
        [(17, 18, 19), (18, 19, 20)]  # Pinky
    ]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    
    for i, finger in enumerate(fingers):
        for j, (p1, p2, p3) in enumerate(finger):
            joint_type = "knuckle" if j == 0 else "middle_joint"
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                features_dict[f"{finger_names[i]}_{joint_type}_angle"] = float(angle)
            else:
                features_dict[f"{finger_names[i]}_{joint_type}_angle"] = 0.0

    fingertips = [4, 8, 12, 16, 20]
    
    for i, tip in enumerate(fingertips):
        dist = np.linalg.norm(landmarks[tip] - palm_center)
        features_dict[f"{finger_names[i]}_tip_to_palm_dist"] = float(dist)
    
    for i, tip in enumerate(fingertips):
        height = landmarks[tip, 1] - wrist[1]
        features_dict[f"{finger_names[i]}_height"] = float(height)
    
    features_dict["thumb_to_index_dist"] = float(np.linalg.norm(landmarks[4] - landmarks[8]))
    features_dict["thumb_to_pinky_dist"] = float(np.linalg.norm(landmarks[4] - landmarks[20]))
    
    avg_fingertip_dist = np.mean([np.linalg.norm(landmarks[tip] - palm_center) for tip in fingertips])
    features_dict["hand_curvature"] = float(avg_fingertip_dist)
    
    spread_distances = []
    for i in range(len(fingertips) - 1):
        dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[i+1]])
        spread_distances.append(dist)
    
    features_dict["finger_spread"] = float(np.mean(spread_distances))
    
    features_dict["thumb_pinky_opposition"] = float(np.linalg.norm(landmarks[4] - landmarks[17]))
    
    palm_normal = np.cross(
        landmarks[5] - landmarks[0],
        landmarks[17] - landmarks[0]
    )
    if np.linalg.norm(palm_normal) > 0:
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        for i, tip in enumerate(fingertips):
            vec_to_tip = landmarks[tip] - landmarks[0]
            dist_to_plane = abs(np.dot(vec_to_tip, palm_normal))
            features_dict[f"{finger_names[i]}_dist_to_palm_plane"] = float(dist_to_plane)
    else:
        for i in range(len(fingertips)):
            features_dict[f"{finger_names[i]}_dist_to_palm_plane"] = 0.0
    
    return features_dict
