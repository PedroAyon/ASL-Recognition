import cv2
import mediapipe as mp
import torch
import numpy as np

# Load the trained model
model_path = "../asl_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(model_path, map_location=device)
model_state = checkpoint["model_state_dict"]
label_to_idx = checkpoint["label_to_idx"]
idx_to_label = checkpoint["idx_to_label"]

num_classes = len(label_to_idx)

# Load the model architecture
class SimpleASLModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleASLModel(input_dim=63, hidden_dim=128, num_classes=num_classes).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)
    label_predicted = "Not sure"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks (flatten to match model input)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            landmarks_np = np.array([coord for point in landmarks for coord in point], dtype=np.float32)

            # Convert to tensor
            landmarks_tensor = torch.tensor(landmarks_np).to(device).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(landmarks_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

            confidence = confidence.item()
            predicted_label = idx_to_label[predicted_idx.item()]

            # Confidence threshold
            if confidence > 0.6:
                label_predicted = predicted_label

    # Display predicted label
    cv2.putText(frame, f"Prediction: {label_predicted}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
