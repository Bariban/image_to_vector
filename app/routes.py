from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms, models
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask приложение
app = Flask(__name__)

# Используем ResNet50
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Трансформации для изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def fetch_image(image_url):
    """Загрузка изображения по URL."""
    try:
        response = requests.get(image_url, timeout=10)  # Ограничение по времени
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при загрузке изображения: {e}")
        return None, f"Failed to fetch image: {str(e)}"
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}")
        return None, f"Invalid image format: {str(e)}"

def process_image(image):
    """Извлечение признаков из изображения."""
    try:
        # Применение трансформаций
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Извлечение признаков
        with torch.no_grad():
            features = model(image_tensor)
        return features.squeeze().cpu().numpy().tolist(), None
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}")
        return None, f"Failed to process image: {str(e)}"

@app.route("/extract_features", methods=["POST"])
def extract_features():
    """API-метод для обработки изображения и получения вектора признаков."""
    image_url = request.form.get("image_url")
    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    # Загрузка изображения
    image, error = fetch_image(image_url)
    if error:
        return jsonify({"error": error}), 400

    # Извлечение признаков
    features, error = process_image(image)
    if error:
        return jsonify({"error": error}), 500

    return jsonify({"features": features})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
