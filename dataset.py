import cv2
import os


gesture_name = 'SMILE'
num_images = 1000
output_dir = f'./dataset/{gesture_name}'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print(f"Capturing images for gesture: {gesture_name}")
print("Press 's' to start capturing and 'q' to quit.")


box_size = 200
box_x = 220
box_y = 140

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break


    frame = cv2.flip(frame, 1)


    cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)


    cv2.imshow('Hand Tracking - Press "s" to start, "q" to quit', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        print("Image capturing cancelled.")
        exit()

count = 0

while count < num_images:

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break


    frame = cv2.flip(frame, 1)


    roi = frame[box_y:box_y + box_size, box_x:box_x + box_size]


    cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)


    cv2.imshow('Hand Tracking - Press "q" to quit', frame)


    img_name = os.path.join(output_dir, f"{gesture_name}_{count:04d}.jpg")
    cv2.imwrite(img_name, roi)
    print(f"Captured {count + 1}/{num_images}: {img_name}")

    count += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Image capturing complete.")
