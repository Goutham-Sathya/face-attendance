import cv2

print("Testing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam with index 0")
    print("Trying index 1...")
    cap = cv2.VideoCapture(1)
    
if cap.isOpened():
    print("SUCCESS: Webcam opened!")
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured: {frame.shape}")
        cv2.imshow('Test', frame)
        cv2.waitKey(3000)  # Show for 3 seconds
    else:
        print("ERROR: Could not read frame")
else:
    print("ERROR: Could not open webcam")

cap.release()
cv2.destroyAllWindows()
print("Test complete")