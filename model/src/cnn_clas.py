import cv2
import torch
from torchvision import transforms
from model.src.cnn import AccidentCNN
import os

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_WEIGHTS_PATH = r"D:\project\bachelor\model\weights\accident_cnn_model.pt"

# ---------------- LOAD MODEL ----------------
cnn_model = AccidentCNN().to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_WEIGHTS_PATH, map_location=DEVICE))
cnn_model.eval()  # важливо для правильної роботи

# ---------------- TRANSFORMS ----------------
cnn_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- PREDICTION FUNCTION ----------------
def classify_image(image_path, threshold=0.65):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Зчитування зображення
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Преобразування для моделі
    tensor = cnn_transforms(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = cnn_model(tensor)
        # Якщо вихід BCE
        if out.shape[1] == 1:
            score = torch.sigmoid(out).item()
        else:
            score = torch.softmax(out, dim=1)[0,1].item()

    # Класифікація
    label = "ACCIDENT" if score > threshold else "NORMAL"
    print(f"[{label}] score={score:.3f}")

    # Опціонально: накладення на зображення
    color = (0,0,255) if label=="ACCIDENT" else (0,255,0)
    cv2.putText(image, f"{label} {score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return label, score

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    classify_image(r"D:\project\bachelor\1.jpg")
