import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# ✅ Define the same CNN model as used during training
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ✅ Class labels as per FER-2013
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Setup device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# ✅ Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Start webcam with DirectShow backend (more reliable on Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Unable to access the camera. Try changing the camera index.")
    exit()

print("🎥 Real-Time Emotion Detection Started — Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            face_tensor = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = classes[predicted.item()]
        except Exception as e:
            emotion = "Error"
            print(f"⚠️ Emotion prediction failed: {e}")

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2)

    cv2.imshow('Real-Time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
