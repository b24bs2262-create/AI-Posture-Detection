import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="posture_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model
    img = cv2.resize(frame, (96, 96))                   # Resize to model size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Convert to grayscale
    img = np.expand_dims(img, axis=-1)                 # Shape (96,96,1)
    img = img.astype('float32') / 255.0                # Normalize
    img = np.expand_dims(img, axis=0)                  # Shape (1,96,96,1)

    # Run prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    prediction = np.argmax(output_data)
    label = "Good Posture" if prediction == 0 else "Bad Posture"

    # Show on screen
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if prediction == 0 else (0, 0, 255), 2)
    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

