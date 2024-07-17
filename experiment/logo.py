import cv2
import numpy as np

# Load the logo image
# Make sure the logo image has an alpha channel (transparency)
logo = cv2.imread('main_logo.png', -1)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get dimensions of the frame
    h_frame, w_frame, _ = frame.shape

    # Get dimensions of the logo
    h_logo, w_logo, _ = logo.shape

    # Define the maximum allowed dimensions for the logo to fit within the frame
    max_logo_width = int(w_frame * 0.2)
    max_logo_height = int(h_frame * 0.2)

    # Resize the logo if it is larger than the maximum allowed dimensions
    if h_logo > max_logo_height or w_logo > max_logo_width:
        scaling_factor = min(max_logo_width / w_logo, max_logo_height / h_logo)
        new_size = (int(w_logo * scaling_factor), int(h_logo * scaling_factor))
        logo = cv2.resize(logo, new_size, interpolation=cv2.INTER_AREA)

        # Update dimensions of the resized logo
        h_logo, w_logo, _ = logo.shape

    # Create a mask for the logo and also create its inverse mask
    bgr_logo = logo[:, :, :3]
    alpha_logo = logo[:, :, 3]

    # Create the inverse mask
    inv_alpha_logo = cv2.bitwise_not(alpha_logo)

    # Position the logo at the top-left corner of the frame
    # You can change these values to position the logo elsewhere
    top_left = (10, 10)
    bottom_right = (top_left[0] + w_logo, top_left[1] + h_logo)

    # Extract the region of interest (ROI) from the frame
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Black out the area of the logo in the ROI
    roi_bg = cv2.bitwise_and(roi, roi, mask=inv_alpha_logo)

    # Take only the region of the logo from the logo image
    roi_fg = cv2.bitwise_and(bgr_logo, bgr_logo, mask=alpha_logo)

    # Add the ROI background and ROI foreground
    dst = cv2.add(roi_bg, roi_fg)

    # Place the combined result back into the original frame
    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = dst

    # Display the frame
    cv2.imshow('Webcam with Logo', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
