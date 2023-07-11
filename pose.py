import cv2
import mediapipe as np
np_drawing = np.solutions.drawing_utils
np_drawing_styles = np.solutions.drawing_styles
np_pose = np.solutions.pose

IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with np_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[np_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[np_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    np_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        np_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=np_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    np_drawing.plot_landmarks(
        results.pose_world_landmarks, np_pose.POSE_CONNECTIONS)

# Entrada de vídeo:
cap = cv2.VideoCapture(0)
with np_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Não está dando vídeo.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    np_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        np_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=np_drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()