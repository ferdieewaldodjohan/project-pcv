import cv2
import mediapipe as mp
import numpy as np
import os
from enum import Enum

class HeadDirection(Enum):
    LEFT = "left"
    FRONT = "front"
    RIGHT = "right"

class BodyDirection(Enum):
    LEFT = "left"
    FRONT = "front"
    RIGHT = "right"

class ArmPose(Enum):
    STRAIGHT = "straight"
    BENT = "bent"

class VTuberSpriteSystem:
    
    def __init__(self):
        print("VTuber Avatar System")
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        self.sprites = {}
        self.TARGET_SIZE = (450, 500)
        self.load_all_sprites()
        
        self.prev_head_rotation = 0.0
        self.prev_x = None
        self.smooth_factor = 0.3
        self.position_smooth = 0.7
        
        self.head_direction = HeadDirection.FRONT
        self.body_direction = BodyDirection.FRONT
        self.left_arm_pose = ArmPose.STRAIGHT
        self.right_arm_pose = ArmPose.STRAIGHT
        
        # Thresholds
        self.HEAD_THRESHOLD = 15
        self.BODY_TURN_THRESHOLD = 0.015  # Y difference threshold
        self.ARM_BEND_THRESHOLD = 110
        
        self.face_model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        
        print("✓ Ready\n")
    
    def load_all_sprites(self):
        sprite_folder = "position"
        sprite_files = {
            'front': 'front.png',
            'lf_face': 'lf_face.png',
            'rg_face': 'rg_face.png',
            'lf_face_body': 'lf_face_body.png',
            'rg_face_body': 'rg_face_body.png',
            'lf_hand': 'lf_hand.png',
            'rg_hand': 'rg_hand.png',
            'bg': 'bg.png'
        }
        
        for name, filename in sprite_files.items():
            filepath = os.path.join(sprite_folder, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    crop_h = int(h * 0.6)
                    img_cropped = img[:crop_h, :]
                    
                    if name != 'bg':
                        img_resized = cv2.resize(img_cropped, self.TARGET_SIZE)
                        self.sprites[name] = img_resized
                    else:
                        self.sprites[name] = img_cropped
    
    def calculate_head_rotation(self, face_landmarks, img_shape):
        h, w = img_shape[:2]
        
        points_2d = np.array([
            [face_landmarks[i].x * w, face_landmarks[i].y * h]
            for i in [1, 152, 263, 33, 291, 61]
        ], dtype=np.float64)
        
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        
        success, rot_vec, _ = cv2.solvePnP(
            self.face_model_points, points_2d, cam_matrix, np.zeros((4, 1))
        )
        
        if success:
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            yaw = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
            self.prev_head_rotation = (
                self.prev_head_rotation * (1 - self.smooth_factor) +
                yaw * self.smooth_factor
            )
        
        return self.prev_head_rotation
    
    def calculate_body_direction(self, pose_landmarks):
        l_sh = pose_landmarks[11]
        r_sh = pose_landmarks[12]
        
        if l_sh.visibility < 0.5 or r_sh.visibility < 0.5:
            return 0.0
        
        # Y difference (smaller Y = higher on screen)
        return l_sh.y - r_sh.y
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        if not all([shoulder.visibility > 0.5, elbow.visibility > 0.5, wrist.visibility > 0.5]):
            return 180
        
        s = np.array([shoulder.x, shoulder.y])
        e = np.array([elbow.x, elbow.y])
        w = np.array([wrist.x, wrist.y])
        
        v1, v2 = s - e, w - e
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    
    def detect_pose_state(self, results, img_shape):
        if not results.pose_landmarks:
            return None
        
        pose = results.pose_landmarks.landmark
        
        # Head
        head_rot = 0.0
        if results.face_landmarks:
            head_rot = self.calculate_head_rotation(results.face_landmarks.landmark, img_shape)
        
        self.head_direction = (
            HeadDirection.LEFT if head_rot < -self.HEAD_THRESHOLD else
            HeadDirection.RIGHT if head_rot > self.HEAD_THRESHOLD else
            HeadDirection.FRONT
        )
        
        # Body - Y coordinate based
        y_diff = self.calculate_body_direction(pose)
        
        self.body_direction = (
            BodyDirection.LEFT if y_diff > self.BODY_TURN_THRESHOLD else
            BodyDirection.RIGHT if y_diff < -self.BODY_TURN_THRESHOLD else
            BodyDirection.FRONT
        )
        
        # Arms
        l_angle = self.calculate_arm_angle(pose[11], pose[13], pose[15])
        r_angle = self.calculate_arm_angle(pose[12], pose[14], pose[16])
        
        self.left_arm_pose = ArmPose.BENT if l_angle < self.ARM_BEND_THRESHOLD else ArmPose.STRAIGHT
        self.right_arm_pose = ArmPose.BENT if r_angle < self.ARM_BEND_THRESHOLD else ArmPose.STRAIGHT
        
        return {
            'head_rotation': head_rot,
            'y_diff': y_diff,
            'left_arm_angle': l_angle,
            'right_arm_angle': r_angle
        }
    
    def select_sprite(self):
        if self.left_arm_pose == ArmPose.BENT:
            return 'lf_hand'
        if self.right_arm_pose == ArmPose.BENT:
            return 'rg_hand'
        
        head = self.head_direction
        body = self.body_direction
        
        if head == HeadDirection.LEFT and body == BodyDirection.LEFT:
            return 'lf_face_body'
        if head == HeadDirection.RIGHT and body == BodyDirection.RIGHT:
            return 'rg_face_body'
        if head == HeadDirection.LEFT:
            return 'lf_face'
        if head == HeadDirection.RIGHT:
            return 'rg_face'
        
        return 'front'
    
    def overlay_sprite(self, canvas, sprite, center_x, bottom_y, scale):
        if sprite is None:
            return
        
        h, w = sprite.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        sprite = cv2.resize(sprite, (new_w, new_h))
        
        x = int(center_x - new_w / 2)
        y = int(bottom_y - new_h)
        
        x = max(0, min(x, canvas.shape[1] - new_w))
        y = max(0, y)
        if y + new_h > canvas.shape[0]:
            new_h = canvas.shape[0] - y
            sprite = sprite[:new_h, :]
        
        sh, sw = sprite.shape[:2]
        
        if sprite.shape[2] == 4:
            alpha = sprite[:, :, 3:4] / 255.0
            fg = sprite[:, :, :3]
            bg = canvas[y:y+sh, x:x+sw]
            canvas[y:y+sh, x:x+sw] = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)
        else:
            canvas[y:y+sh, x:x+sw] = sprite
    
    def draw_keypoints(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return frame
    
    def render(self, canvas, pose_landmarks, w, h):
        if not pose_landmarks:
            return
        
        pose = pose_landmarks
        l_hip, r_hip = pose[23], pose[24]
        
        if l_hip.visibility < 0.3 or r_hip.visibility < 0.3:
            return
        
        center_x = int((l_hip.x + r_hip.x) / 2 * w)
        if self.prev_x is None:
            self.prev_x = center_x
        else:
            self.prev_x = int(self.prev_x * self.position_smooth + center_x * (1 - self.position_smooth))
        
        bottom_y = h - 50
        scale = (w * 0.35) / self.TARGET_SIZE[0]
        
        sprite_name = self.select_sprite()
        sprite = self.sprites.get(sprite_name)
        self.overlay_sprite(canvas, sprite, self.prev_x, bottom_y, scale)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        bg = self.sprites.get('bg')
        show_stats = True
        
        print("Press 'q' quit, 's' toggle stats\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            canvas = (cv2.resize(bg[:, :, :3], (w, h)) if bg is not None 
                     else np.ones((h, w, 3), dtype=np.uint8) * 50)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            
            debug_frame = frame.copy()
            debug_frame = self.draw_keypoints(debug_frame, results)
            
            if results.pose_landmarks:
                stats = self.detect_pose_state(results, frame.shape)
                self.render(canvas, results.pose_landmarks.landmark, w, h)
                
                # Debug display
                if stats:
                    y_diff = stats['y_diff']
                    y = 30
                    
                    if abs(y_diff) < self.BODY_TURN_THRESHOLD:
                        color = (0, 255, 0)
                        status = "FRONT"
                    elif y_diff > 0:
                        color = (255, 100, 255)
                        status = "LEFT"
                    else:
                        color = (100, 200, 255)
                        status = "RIGHT"
                    
                    cv2.putText(debug_frame, f"Y Diff: {y_diff:.4f}", 
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    y += 40
                    cv2.putText(debug_frame, f"Threshold: +/-{self.BODY_TURN_THRESHOLD:.3f}", 
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    y += 40
                    cv2.putText(debug_frame, f"Body: {status}", 
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # Stats overlay
                if show_stats and stats:
                    cv2.rectangle(canvas, (10, 10), (380, 190), (0, 0, 0), -1)
                    cv2.rectangle(canvas, (10, 10), (380, 190), (0, 255, 0), 2)
                    
                    y = 35
                    cv2.putText(canvas, "POSE DETECTION", (20, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 35
                    
                    head_col = (100, 200, 255) if self.head_direction != HeadDirection.FRONT else (255, 255, 255)
                    body_col = (255, 100, 100) if self.body_direction != BodyDirection.FRONT else (255, 255, 255)
                    
                    cv2.putText(canvas, f"Head: {self.head_direction.value.upper()} ({stats['head_rotation']:.1f})", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_col, 2)
                    y += 30
                    cv2.putText(canvas, f"Body: {self.body_direction.value.upper()} (y: {stats['y_diff']:.3f})", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, body_col, 2)
                    y += 30
                    
                    l_col = (255, 200, 100) if self.left_arm_pose == ArmPose.BENT else (255, 255, 255)
                    r_col = (255, 200, 100) if self.right_arm_pose == ArmPose.BENT else (255, 255, 255)
                    
                    cv2.putText(canvas, f"L Arm: {self.left_arm_pose.value.upper()} ({stats['left_arm_angle']:.0f})", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_col, 1)
                    y += 25
                    cv2.putText(canvas, f"R Arm: {self.right_arm_pose.value.upper()} ({stats['right_arm_angle']:.0f})", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_col, 1)
                    y += 30
                    
                    cv2.putText(canvas, f"Active: {self.select_sprite()}", 
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            else:
                cv2.putText(canvas, "NO TRACKING", (w//2 - 150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(debug_frame, "NO TRACKING", (w//2 - 150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.imshow('VTuber Avatar', canvas)
            cv2.imshow('MediaPipe Keypoints', debug_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('s'):
                show_stats = not show_stats
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = VTuberSpriteSystem()
    system.run()