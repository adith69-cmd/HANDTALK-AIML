from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import cv2
import numpy as np

# Load the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define the actions
actions = ['hii', 'thanks', 'noo']  # Your predefined actions
colors = [(245, 117, 16)] * len(actions)  # Colors for visualization

print(len(colors))

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame

# New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8 

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try: 
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action_index = np.argmax(res)
                predicted_action = actions[predicted_action_index]

                # Check if the prediction is valid
                if res[predicted_action_index] > threshold:
                    print(predicted_action)
                    predictions.append(predicted_action_index)
                else:
                    print("Invalid symbol")
                    predictions.append(-1)  # Append -1 for invalid actions
                
            # 3. Viz logic
            if len(predictions) > 0 and np.unique(predictions[-10:])[0] != -1:
                if len(sentence) > 0 and predictions[-1] != sentence[-1]:
                    sentence.append(predictions[-1])
                    accuracy.append(str(res[predicted_action_index] * 100))
                else:
                    sentence.append(predictions[-1])
                    accuracy.append(str(res[predicted_action_index] * 100)) 

            if len(sentence) > 1: 
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

        except Exception as e:
            pass
            
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        output_text = "Output: -" + ' '.join(actions[i] for i in sentence if i != -1)  # Filter out invalid actions
        output_text += ' ' + ''.join(accuracy)
        cv2.putText(frame, output_text, (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
