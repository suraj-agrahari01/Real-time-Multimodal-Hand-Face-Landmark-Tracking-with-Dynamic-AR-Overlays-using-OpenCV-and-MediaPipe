import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Load the logo image
logo = cv2.imread('dr_strange.png', -1)


def detect_hand_landmarks_and_place_logo(image):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image and place the logo
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
            #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
            h_logo, w_logo, _ = logo.shape

            if h_logo > palm_height or w_logo > palm_width:
                scaling_factor = min(palm_width / w_logo, palm_height / h_logo)
                new_size = (int(w_logo * scaling_factor * 4),
                            int(h_logo * scaling_factor * 4))
                resized_logo = cv2.resize(
                    logo, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_logo = logo

            # Update dimensions of the resized logo
            h_logo, w_logo, _ = resized_logo.shape

            # Create a mask for the logo and also create its inverse mask
            bgr_logo = resized_logo[:, :, :3]
            alpha_logo = resized_logo[:, :, 3]

            # Create the inverse mask
            inv_alpha_logo = cv2.bitwise_not(alpha_logo)

            # Position the logo at the center point
            top_left = (center_x - w_logo // 2, center_y - h_logo // 2)
            bottom_right = (top_left[0] + w_logo, top_left[1] + h_logo)

            # Make sure the logo is within the image bounds
            top_left = (max(top_left[0], 0), max(top_left[1], 0))
            bottom_right = (min(bottom_right[0], w), min(bottom_right[1], h))

            # Extract the region of interest (ROI) from the frame
            roi = image[top_left[1]:bottom_right[1],
                        top_left[0]:bottom_right[0]]

            # Check if the ROI size matches the logo size; if not, adjust the logo size
            roi_h, roi_w = roi.shape[:2]
            if roi_h != h_logo or roi_w != w_logo:
                resized_logo = cv2.resize(
                    resized_logo, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
                bgr_logo = resized_logo[:, :, :3]
                alpha_logo = resized_logo[:, :, 3]
                inv_alpha_logo = cv2.bitwise_not(alpha_logo)

            # Black out the area of the logo in the ROI
            roi_bg = cv2.bitwise_and(roi, roi, mask=inv_alpha_logo)

            # Take only the region of the logo from the logo image
            roi_fg = cv2.bitwise_and(bgr_logo, bgr_logo, mask=alpha_logo)

            # Add the ROI background and ROI foreground
            dst = cv2.add(roi_bg, roi_fg)

            # Place the combined result back into the original frame
            image[top_left[1]:bottom_right[1],
                  top_left[0]:bottom_right[0]] = dst

    return image


def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Detect hand landmarks and place logo
        frame = detect_hand_landmarks_and_place_logo(frame)

        # Resize the frame to 900x900
        frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)

        # Display the frame
        cv2.imshow('Hand Landmarks with Logo', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
