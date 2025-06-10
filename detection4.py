import cv2
import numpy as np

def detect_disease_from_yellow_shade(hsv_color):
    h, s, v = hsv_color
    if v > 180 and s < 100:
        return "Nitrogen Deficiency (light yellow)"
    elif s > 150 and v < 180:
        return "Fungal Infection (dark yellow)"
    elif s > 100 and v > 180:
        return "Potassium Deficiency (bright yellow)"
    else:
        return "Unknown yellow shade"

def nothing(x):
    pass

def enhance_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Create trackbars window
cv2.namedWindow("Controls")
cv2.createTrackbar("H_min", "Controls", 20, 179, nothing)
cv2.createTrackbar("H_max", "Controls", 35, 179, nothing)
cv2.createTrackbar("S_min", "Controls", 50, 255, nothing)
cv2.createTrackbar("S_max", "Controls", 255, 255, nothing)
cv2.createTrackbar("V_min", "Controls", 50, 255, nothing)
cv2.createTrackbar("V_max", "Controls", 255, 255, nothing)
cv2.createTrackbar("MinArea", "Controls", 500, 5000, nothing)
cv2.createTrackbar("Brightness", "Controls", 0, 100, nothing)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
cap.set(cv2.CAP_PROP_GAIN, 1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    brightness = cv2.getTrackbarPos("Brightness", "Controls")
    if brightness > 0:
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)

    enhanced = enhance_low_light(frame)

    h_min = cv2.getTrackbarPos("H_min", "Controls")
    h_max = cv2.getTrackbarPos("H_max", "Controls")
    s_min = cv2.getTrackbarPos("S_min", "Controls")
    s_max = cv2.getTrackbarPos("S_max", "Controls")
    v_min = cv2.getTrackbarPos("V_min", "Controls")
    v_max = cv2.getTrackbarPos("V_max", "Controls")
    min_area = cv2.getTrackbarPos("MinArea", "Controls")

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

    lower_yellow = np.array([h_min, s_min, v_min])
    upper_yellow = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if large_contours:
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = hsv[y:y+h, x:x+w]

            if roi.size > 0:
                avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                disease = detect_disease_from_yellow_shade(avg_color)

                cv2.rectangle(enhanced, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(enhanced, disease, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(enhanced, f"H:{avg_color[0]:.1f} S:{avg_color[1]:.1f} V:{avg_color[2]:.1f}",
                            (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(enhanced, "No yellow detected - adjust thresholds", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced", enhanced)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
