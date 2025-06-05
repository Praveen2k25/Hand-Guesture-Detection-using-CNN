import cv2
import numpy as np
from keras.models import load_model


model = load_model("Hand-Guesture-Detection-using-CNN\hand_gesture_model_gray.h5")


classes = ['open hand', 'closed hand'] 
unknown_threshold = 0.5  


def preprocess_roi(roi):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
    roi_resized = cv2.resize(roi_gray, (64, 64))  
    roi_reshaped = np.reshape(roi_resized, (64, 64, 1))  
    roi_normalized = roi_reshaped / 255.0  
    return np.expand_dims(roi_normalized, axis=0)  

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

   
    x, y, w, h = 200, 100, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi = frame[y:y + h, x:x + w]


    roi_processed = preprocess_roi(roi)

    prediction = model.predict(roi_processed)
    predicted_class = np.argmax(prediction)
    predicted_probability = np.max(prediction)  
  
    if predicted_probability < unknown_threshold:
        predicted_label = 'unknown'
    else:
        predicted_label = classes[predicted_class]  

    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow('Hand Gesture Recognition', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
