from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms, models
import logging
from pyzbar.pyzbar import decode, ZBarSymbol

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Конфигурация
CONFIG = {
    "barcode": {
        "resize_width": 640,
        "min_width_ratio": 0.3,
        "min_height_ratio": 0.2,
        "sobel_threshold": 127,
        "morph_kernel": (5, 30),
        "allowed_types": [
            ZBarSymbol.EAN13,    # Стандартные штрихкоды товаров
            ZBarSymbol.EAN8,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
            ZBarSymbol.CODE128,  # Штрихкоды для логистики
        ]
    },
    "model": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    }
}

# Инициализация модели
model = models.resnet50(pretrained=True).to(CONFIG["model"]["device"])
model.eval()

# Трансформации изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(CONFIG["model"]["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG["model"]["normalize_mean"],
        std=CONFIG["model"]["normalize_std"]
    )
])

def fetch_image(image_url: str) -> tuple:
    """Загрузка изображения по URL."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB"), None
    except Exception as e:
        logger.error(f"Image fetch error: {str(e)}")
        return None, f"Image processing failed: {str(e)}"

def preprocess_for_barcode(np_image: np.ndarray) -> np.ndarray:
    """Предобработка изображения для детекции штрихкода."""
    # Уменьшение размера
    h, w = np_image.shape[:2]
    if w > CONFIG["barcode"]["resize_width"]:
        ratio = CONFIG["barcode"]["resize_width"] / w
        np_image = cv2.resize(np_image, (CONFIG["barcode"]["resize_width"], int(h * ratio)))
    
    # Конвертация в grayscale
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    
    # Улучшение контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def fast_barcode_detection(np_image: np.ndarray) -> bool:
    """Быстрая предварительная проверка на наличие штрихкода."""
    try:
        gray = preprocess_for_barcode(np_image)
        
        # Детекция вертикальных границ
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        _, thresh = cv2.threshold(np.absolute(grad_x), CONFIG["barcode"]["sobel_threshold"], 255, cv2.THRESH_BINARY)
        thresh = cv2.convertScaleAbs(thresh)

        # Морфологические операции
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            CONFIG["barcode"]["morph_kernel"]
        )
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Поиск контуров
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w > CONFIG["barcode"]["min_width_ratio"] * gray.shape[1] and 
                h > CONFIG["barcode"]["min_height_ratio"] * gray.shape[0]):
                return True
        return False
    except Exception as e:
        logger.error(f"Barcode detection error: {str(e)}")
        return False

def decode_barcode(image: Image.Image) -> str:
    """Распознавание только штрихкодов (без QR-кодов)"""
    try:
        np_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if not fast_barcode_detection(np_image):
            return None

        # Фильтрация по типам кодов
        decoded = decode(
            image, 
            symbols=CONFIG["barcode"]["allowed_types"]
        )

        return decoded[0].data.decode() if decoded else None
    except Exception as e:
        logger.error(f"Barcode decoding error: {str(e)}")
        return None

def extract_features(image: Image.Image) -> list:
    """Извлечение признаков изображения с помощью ResNet."""
    try:
        image_tensor = transform(image).unsqueeze(0).to(CONFIG["model"]["device"])
        with torch.no_grad():
            features = model(image_tensor)
        return features.squeeze().cpu().numpy().tolist(), None
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        return None, str(e)

@app.route("/extract_features", methods=["POST"])
def handle_request():
    """Основной обработчик запросов."""
    # Валидация входных данных
    if "image_url" not in request.form:
        return jsonify({"error": "Missing image_url parameter"}), 400
    
    image_url = request.form["image_url"]
    
    # Загрузка изображения
    image, error = fetch_image(image_url)
    if error:
        return jsonify({"error": error}), 400

    # Попытка распознать штрихкод
    barcode = decode_barcode(image)
    if barcode:
        logger.info(f"Successfully decoded barcode: {barcode}")
        return jsonify({"barcode": barcode})

    # Извлечение признаков
    features, error = extract_features(image)
    if error:
        return jsonify({"error": error}), 500

    return jsonify({"features": features})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)