import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.5)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Load the logos
spiderman_logo = cv2.imread(
    'assests\spider_man_mask.png', cv2.IMREAD_UNCHANGED)
dr_strange_logo = cv2.imread('assests\dr_strange.png', -1)

# Define button coordinates (top-left and bottom-right corners)
button_coords = [
    [(10, 10), (200, 75)],  # Button 1
    [(220, 10), (410, 75)],  # Button 2
    [(430, 10), (620, 75)],  # Clear Button
]

# Button labels
button_labels = [
    'Dr. Strange',
    'Spider man',
    'Clear',
]

# Flags to indicate if buttons are pressed
button_1_pressed = False
button_2_pressed = False
mode = None


def draw_buttons(image):
    for coords, label in zip(button_coords, button_labels):
        cv2.rectangle(image, coords[0], coords[1], (0, 0, 255), -1)
        cv2.putText(image, label, (coords[0][0] + 10, coords[0][1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def detect_button_press(landmark, button_coords, w, h):
    x, y = int(landmark.x * w), int(landmark.y * h)
    return button_coords[0][0] < x < button_coords[1][0] and button_coords[0][1] < y < button_coords[1][1]


def detect_hand_landmarks(image, w, h):
    global button_1_pressed, button_2_pressed, mode
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image and check for button presses
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if index fingertip (landmark 8) is pressing any button
            index_fingertip = hand_landmarks.landmark[8]

            if detect_button_press(index_fingertip, button_coords[0], w, h):
                button_1_pressed = True
                button_2_pressed = False
                mode = 'Dr. Strange'
            elif detect_button_press(index_fingertip, button_coords[1], w, h):
                button_1_pressed = False
                button_2_pressed = True
                mode = 'Spider man'
            elif detect_button_press(index_fingertip, button_coords[2], w, h):
                button_1_pressed = False
                button_2_pressed = False
                mode = None

    return image


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """ Overlay img_overlay on top of img at (x, y) and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the same size as img_overlay.
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to overlay
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

    return img


def detect_faces(image, w, h):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect faces
    results = face_detection.process(image_rgb)

    # Draw face landmarks on the image
    if results.detections:
        for detection in results.detections:
            for id, landmark in enumerate(detection.location_data.relative_keypoints):
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image, f'{id}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if button_2_pressed:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                int(bboxC.width * iw), int(bboxC.height * ih))

                # Resize the spiderman logo to fit the face bounding box
                spiderman_logo_resized = cv2.resize(
                    spiderman_logo, (w, h + 50))

                # Split the channels
                b, g, r, a = cv2.split(spiderman_logo_resized)

                # Merge the BGR channels and create an alpha mask
                overlay_img = cv2.merge((b, g, r))
                alpha_mask = a / 255.0

                # Overlay the spiderman logo on the face
                image = overlay_image_alpha(
                    image, overlay_img, x, y-20, alpha_mask)

    return image


def detect_hand_landmarks_and_place_logo(image):
    global mode
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image and place the logo
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if mode == 'Dr. Strange':
                # Get the coordinates of landmarks 0, 5, and 17
                landmark_0 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                landmark_5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                landmark_17 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                h, w, _ = image.shape
                l0_x, l0_y = int(landmark_0.x * w), int(landmark_0.y * h)
                l5_x, l5_y = int(landmark_5.x * w), int(landmark_5.y * h)
                l17_x, l17_y = int(landmark_17.x * w), int(landmark_17.y * h)

                # Calculate the center point between landmarks 5 and 17
                center_x = (l5_x + l17_x) // 2
                center_y = (l5_y + l17_y) // 2

                # Calculate the maximum allowed dimensions for the logo to fit within the palm
                palm_width = int(abs(l5_x - l17_x))
                palm_height = int(abs(l0_y - center_y) * 2)

                # Get dimensions of the logo
                h_logo, w_logo, _ = dr_strange_logo.shape

                if h_logo > palm_height or w_logo > palm_width:
                    scaling_factor = min(
                        palm_width / w_logo, palm_height / h_logo)
                    new_size = (int(w_logo * scaling_factor * 4),
                                int(h_logo * scaling_factor * 4))
                    resized_logo = cv2.resize(
                        dr_strange_logo, new_size, interpolation=cv2.INTER_AREA)
                else:
                    resized_logo = dr_strange_logo

                # Update dimensions of the resized logo
                h_logo, w_logo, _ = resized_logo.shape

                # Calculate top-left corner coordinates for the logo overlay
                top_left_x = center_x - w_logo // 2
                top_left_y = center_y - h_logo // 2

                # Split the channels
                b, g, r, a = cv2.split(resized_logo)

                # Merge the BGR channels and create an alpha mask
                overlay_img = cv2.merge((b, g, r))
                alpha_mask = a / 255.0

                # Overlay the resized logo on the image
                image = overlay_image_alpha(
                    image, overlay_img, top_left_x, top_left_y, alpha_mask)

    return image


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        image = cv2.resize(image, (700, 700))

        h, w, _ = image.shape

        # Draw buttons
        draw_buttons(image)

        # Detect hand landmarks and check for button presses
        image = detect_hand_landmarks(image, w, h)

        # Detect faces and add overlay if required
        image = detect_faces(image, w, h)

        # Detect hand landmarks and place the logo
        image = detect_hand_landmarks_and_place_logo(image)

        # Display the resulting frame
        cv2.imshow('MediaPipe Hand and Face Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
