import cv2
import os

for i in range(1, 7):
    os.makedirs(f'data/{i}', exist_ok=True)

cap = cv2.VideoCapture(0)
current_sign = 1
num_images = 200

print(f"Collecting images for sign {current_sign}. Press 'c' to capture, 'q' to quit, or 'n' to move to the next sign.")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('c'):
        img_name = f"data/{current_sign}/{len(os.listdir(f'data/{current_sign}')) + 1}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
    
    elif key == ord('n'):
        current_sign += 1
        if current_sign > 6:
            break
        print(f"Collecting images for sign {current_sign}.")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
