import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import plotly.express as px
import pandas as pd
import os

# ------------------------------------------------------------
# Sayfa konfigürasyonu
# ------------------------------------------------------------
st.set_page_config(
    page_title="ISO 2409 Çapraz Kesim Sınıflandırıcısı",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CSS stilleri
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid;
        text-align: center;
        margin: 1rem 0;
    }
    .preprocessing-steps {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .class-0 { border-color: #27ae60; }
    .class-1 { border-color: #2ecc71; }
    .class-2 { border-color: #f1c40f; }
    .class-3 { border-color: #e67e22; }
    .class-4 { border-color: #e74c3c; }
    .class-5 { border-color: #c0392b; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def draw_square_overlay(pil_img: Image.Image, cx: int, cy: int, size: int, color=(0,255,0), thickness=3) -> np.ndarray:
    img = pil_to_rgb_np(pil_img).copy()
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w - 1, cx + half), min(h - 1, cy + half)
    bgr = img[:, :, ::-1].copy()
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (color[2], color[1], color[0]), thickness)
    return bgr[:, :, ::-1]

def crop_square(pil_img: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    img = pil_to_rgb_np(pil_img)
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cropped)

def make_cut_mask(grid_gray: np.ndarray, thickness_px: int) -> np.ndarray:
    """Kesik çizgilerini maskele: çizgileri bul, kalınlaştır, 0/255 maske üret."""
    edges = cv2.Canny(grid_gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                            minLineLength=max(20, grid_gray.shape[1] // 3),
                            maxLineGap=10)
    mask = np.zeros_like(grid_gray, dtype=np.uint8)
    if lines is not None:
        for l in lines[:, 0]:
            cv2.line(mask, (l[0], l[1]), (l[2], l[3]), 255, max(1, int(thickness_px)))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask

def make_flake_mask_rgb(grid_rgb: np.ndarray, surface_type: str) -> np.ndarray:
    """
    Adaptif flake maskesi:
    - HSV/V kanalında Otsu ile hem 'parlak' hem 'koyu' eşik.
    - Beyaz zeminde koyuyu, koyu/renkli zeminde parlak flake'i tercih et.
    """
    hsv = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]

    _, dark = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bright = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if surface_type == "white":
        flake = dark
    else:
        flake = bright

    flake = cv2.morphologyEx(flake, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return flake

# ------------------------------------------------------------
# Ön işleme adımları gösterimi
# ------------------------------------------------------------
def show_preprocessing_steps(preprocessing_steps):
    st.markdown("""
    <div class="preprocessing-steps">
        <h3>🔄 Görüntü Ön İşleme Adımları</h3>
        <p>5x5 grid yapısı için optimize edilmiş işleme pipeline'ı</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### İlk Satır")
    col1, col2, col3, col4 = st.columns(4)
    if 'original' in preprocessing_steps:
        with col1:
            img = preprocessing_steps['original']; st.image(cv2.resize(img, (250, 250)), caption="Orijinal")
    if 'grayscale' in preprocessing_steps:
        with col2:
            img = preprocessing_steps['grayscale']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (250, 250)), caption="Gri Tonlama")
    if 'denoised' in preprocessing_steps:
        with col3:
            img = preprocessing_steps['denoised']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (250, 250)), caption="Gürültü Azaltılmış")
    if 'contrast_enhanced' in preprocessing_steps:
        with col4:
            img = preprocessing_steps['contrast_enhanced']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (250, 250)), caption="Kontrast Artırılmış")
    st.markdown("### İkinci Satır")
    col1, col2, col3, col4, col5 = st.columns(5)
    if 'bilateral' in preprocessing_steps:
        with col1:
            img = preprocessing_steps['bilateral']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (200, 200)), caption="Kenar Korumalı Filtre")
    if 'morphology' in preprocessing_steps:
        with col2:
            img = preprocessing_steps['morphology']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (200, 200)), caption="Morfolojik İşlem")
    if 'adaptive_threshold' in preprocessing_steps:
        with col3:
            img = preprocessing_steps['adaptive_threshold']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (200, 200)), caption="Adaptif Eşikleme")
    if 'grid_lines' in preprocessing_steps:
        with col4:
            img = preprocessing_steps['grid_lines']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (200, 200)), caption="Grid Çizgileri")
    if 'final_processed' in preprocessing_steps:
        with col5:
            img = preprocessing_steps['final_processed']; st.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (200, 200)), caption="Final İşlenmiş")

# ------------------------------------------------------------
# Sınıflandırıcı sınıfı
# ------------------------------------------------------------
class CrosscutClassifier:
    def __init__(self):
        self.iso_classes = {
            0: {"name": "Sınıf 0","description": "Kesilerin kesişim noktalarında bozulma yok","criteria": "Mükemmel yapışma, hiç ayrılma yok","quality": "Mükemmel","color": "#27ae60"},
            1: {"name": "Sınıf 1","description": "Kesişim noktalarında çok küçük pullar","criteria": "Sadece kesişim noktalarında minimal ayrılma","quality": "Çok İyi","color": "#2ecc71"},
            2: {"name": "Sınıf 2","description": "Kesişim noktalarında ve/veya kesiler boyunca küçük pullar","criteria": "Kesim kenarları boyunca küçük ayrılmalar","quality": "İyi","color": "#f1c40f"},
            3: {"name": "Sınıf 3","description": "Büyük pullar ve/veya kesim kenarları boyunca büyük pullar","criteria": "Karelere doğru uzanan büyük ayrılmalar","quality": "Kabul Edilebilir","color": "#e67e22"},
            4: {"name": "Sınıf 4","description": "Büyük pullar. Çapraz kesim alanının %5'den fazla kısmı bozulur","criteria": "Önemli alan etkilenmiş, belirgin ayrılma","quality": "Zayıf","color": "#e74c3c"},
            5: {"name": "Sınıf 5","description": "Herhangi bir derece, sıkıntılanma pulları","criteria": "Çok zayıf yapışma, yaygın ayrılma","quality": "Çok Zayıf","color": "#c0392b"}
        }
        self.model = None
        self.using_demo_model = True
        self.load_model()

    def load_model(self):
        """
        Eğitimli model varsa kullan, yoksa demo modele düş.
        """
        try:
            model_path = "model_iso2409.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.using_demo_model = False
            else:
                self.model = self.create_demo_model()
                self.using_demo_model = True
        except Exception as e:
            st.warning(f"Eğitimli model yüklenemedi, demo modele düşüldü: {e}")
            self.model = self.create_demo_model()
            self.using_demo_model = True
        return True

    def create_demo_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # ------------------------- ÖN İŞLEME -------------------------
    def enhanced_preprocessing(self, image):
        if isinstance(image, Image.Image):
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
        else:
            image_array = image

        preprocessing_steps = {}
        original = image_array.copy()
        h, w = original.shape[:2]
        if h != w:
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            original = cv2.copyMakeBorder(original, pad_h, pad_h, pad_w, pad_w,
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
        original = cv2.resize(original, (400, 400))
        preprocessing_steps['original'] = original

        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        preprocessing_steps['grayscale'] = gray
        denoised = cv2.fastNlMeansDenoising(gray)
        preprocessing_steps['denoised'] = denoised
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        preprocessing_steps['contrast_enhanced'] = contrast_enhanced
        bilateral = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
        preprocessing_steps['bilateral'] = bilateral
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(bilateral, cv2.MORPH_OPEN, kernel)
        preprocessing_steps['morphology'] = opening
        adaptive_thresh = cv2.adaptiveThreshold(
            opening, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessing_steps['adaptive_threshold'] = adaptive_thresh
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel)
        grid_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        preprocessing_steps['grid_lines'] = grid_lines
        final_processed = cv2.addWeighted(opening, 0.8, grid_lines, 0.2, 0)
        preprocessing_steps['final_processed'] = final_processed

        processed_rgb = cv2.cvtColor(final_processed, cv2.COLOR_GRAY2RGB)
        processed_rgb = cv2.resize(processed_rgb, (224, 224)).astype(np.float32) / 255.0
        model_input = np.expand_dims(processed_rgb, axis=0)
        return model_input, preprocessing_steps

    # ------------------------- GRID BÖLGESİ -------------------------
    def detect_crosscut_grid_region(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        if lines is None:
            h, w = gray.shape
            return {'x': w//4, 'y': h//4, 'width': w//2, 'height': h//2, 'detected': False}
        horizontal_lines, vertical_lines = [], []
        for line in lines:
            rho, theta = line[0]
            if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                horizontal_lines.append(rho)
            elif abs(theta - np.pi/2) < np.pi/4:
                vertical_lines.append(rho)
        if len(horizontal_lines) < 4 or len(vertical_lines) < 4:
            h, w = gray.shape
            return {'x': w//4, 'y': h//4, 'width': w//2, 'height': h//2, 'detected': False}
        horizontal_lines.sort(); vertical_lines.sort()
        min_x = int(abs(min(vertical_lines))); max_x = int(abs(max(vertical_lines)))
        min_y = int(abs(min(horizontal_lines))); max_y = int(abs(max(horizontal_lines)))
        grid_width = max_x - min_x; grid_height = max_y - min_y
        return {
            'x': max(0, min_x - 20),
            'y': max(0, min_y - 20),
            'width': min(gray.shape[1] - max(0, min_x - 20), grid_width + 40),
            'height': min(gray.shape[0] - max(0, min_y - 20), grid_height + 40),
            'detected': True
        }

    # ------------------------- 5x5 Analizi (kesikler maskeli) -------------------------
    def analyze_5x5_grid_original(
        self,
        original_image,
        spacing_mm: int = 1,
        strict_cell_damage: bool = True,
        flake_ratio_thr: float = 0.003,
        min_pix_ratio: float = 0.001,
        return_debug: bool = False
    ):
        """Yalnızca hücre içi kopmaları say (kesik çizgileri hariç). spacing_mm: 1/2/3."""
        # PIL -> NumPy
        if isinstance(original_image, Image.Image):
            if original_image.mode == 'RGBA':
                bg = Image.new('RGB', original_image.size, (255, 255, 255))
                bg.paste(original_image, mask=original_image.split()[-1])
                original_image = bg
            elif original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            work_image = np.array(original_image)
        else:
            work_image = original_image

        # Gri + grid bölgesi
        gray_full = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY) if len(work_image.shape) == 3 else work_image
        grid_region = self.detect_crosscut_grid_region(work_image)
        x, y, w, h = grid_region['x'], grid_region['y'], grid_region['width'], grid_region['height']
        grid_gray = gray_full[y:y+h, x:x+w]
        grid_rgb  = work_image[y:y+h, x:x+w] if len(work_image.shape) == 3 else cv2.cvtColor(grid_gray, cv2.COLOR_GRAY2RGB)

        # Grid kalite skoru (opsiyonel)
        edges = cv2.Canny(grid_gray, 30, 100)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        grid_quality_score = 0
        if lines is not None:
            horiz, vert = [], []
            for line in lines:
                rho, theta = line[0]
                if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                    horiz.append(line)
                elif abs(theta - np.pi/2) < np.pi/4:
                    vert.append(line)
            grid_quality_score = min(len(horiz), len(vert)) / 6 * 100

        # Yüzey tipi (kabaca)
        mean_r, mean_g, mean_b = np.mean(grid_rgb[:,:,0]), np.mean(grid_rgb[:,:,1]), np.mean(grid_rgb[:,:,2])
        if mean_r > mean_g and mean_r > mean_b and mean_r > 150:
            surface_type = "red"
        elif mean_g > mean_r and mean_g > mean_b and mean_g > 150:
            surface_type = "green"
        elif mean_b > mean_r and mean_b > mean_g and mean_b > 150:
            surface_type = "blue"
        elif mean_r > 200 and mean_g > 200 and mean_b > 200:
            surface_type = "white"
        else:
            surface_type = "dark"

        # Hücre boyutu ve piksel/mm tahmini
        height, width = grid_gray.shape
        cell_h = max(1, height // 5); cell_w = max(1, width // 5)
        px_per_mm = max(1.0, ((cell_w + cell_h) / 2.0) / float(spacing_mm))

        # Dinamik eşikler
        thickness_px = int(np.clip(round(0.35 * px_per_mm), 2, 14))  # kesik kalınlığı maskesi
        cell_area = cell_w * cell_h
        if strict_cell_damage:
            FLAKE_RATIO_THR = float(flake_ratio_thr)     # varsayılan %0.3
            MIN_PIX = max(5, int(min_pix_ratio * cell_area))
        else:
            FLAKE_RATIO_THR = max(float(flake_ratio_thr), 0.01)
            MIN_PIX = max(30, int(max(min_pix_ratio, 0.005) * cell_area))

        # 1) Kesik çizgisi maskesi
        cut_mask = make_cut_mask(grid_gray, thickness_px)
        interior_mask = cv2.bitwise_not(cut_mask)

        # 2) Flake maskesi (adaptif)
        flake_mask = make_flake_mask_rgb(grid_rgb, surface_type)

        # 3) Yalnızca kesik dışı flake
        flake_mask_interior = cv2.bitwise_and(flake_mask, interior_mask)

        # Hücre bazında sayım
        total_damaged_cells = 0.0
        cell_damage_scores = []
        interior_total = int(np.sum(interior_mask > 0))
        flake_interior_total = int(np.sum(flake_mask_interior > 0))

        for i in range(5):
            for j in range(5):
                y1 = i * cell_h; y2 = min(height, (i + 1) * cell_h)
                x1 = j * cell_w;  x2 = min(width,  (j + 1) * cell_w)
                cell_interior = interior_mask[y1:y2, x1:x2]
                cell_flake    = flake_mask_interior[y1:y2, x1:x2]

                area = max(1, int(np.sum(cell_interior > 0)))
                flake_pix = int(np.sum(cell_flake > 0))
                ratio = flake_pix / area

                damage_score = 1.0 if (flake_pix >= MIN_PIX and ratio >= FLAKE_RATIO_THR) else 0.0
                cell_damage_scores.append(float(damage_score))
                total_damaged_cells += damage_score

        # Ayrılma oranı: kesik hariç grid alanında flake yüzdesi
        delamination_ratio = 100.0 * (flake_interior_total / interior_total) if interior_total > 0 else 0.0

        result = {
            'grid_quality_score': float(grid_quality_score),
            'delamination_ratio': float(delamination_ratio),
            'damaged_cells': float(total_damaged_cells),  # 0..25
            'total_cells': 25,
            'damage_percentage': (total_damaged_cells / 25.0) * 100.0,
            'cell_damage_scores': [float(x) for x in cell_damage_scores],
            'grid_detected': bool(lines is not None and len(lines) > 8),
            'analysis_method': f'Grid Region Analysis - {surface_type} (cuts masked)',
            'surface_type': surface_type,
            'grid_region': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'spacing_mm': spacing_mm,
            'px_per_mm': px_per_mm,
            'cut_thickness_px': thickness_px,
            'cell_area_px': cell_area,
            'flake_ratio_thr': FLAKE_RATIO_THR,
            'min_pix': MIN_PIX
        }

        if return_debug:
            result['debug_masks'] = {
                'cut_mask': cut_mask,
                'flake_mask': flake_mask,
                'flake_mask_interior': flake_mask_interior
            }

        return result

    # ------------------------- PREDICT -------------------------
    def predict(
        self,
        image,
        spacing_mm: int = 1,
        strict_cell_damage: bool = True,
        flake_ratio_thr: float = 0.003,
        min_pix_ratio: float = 0.001,
        return_debug: bool = False
    ):
        if self.model is None:
            return None
        st.info("ADIM 1: Görüntü ön işleme başlıyor...")
        processed_input, preprocessing_steps = self.enhanced_preprocessing(image)
        st.success("Ön işleme tamamlandı!")
        show_preprocessing_steps(preprocessing_steps)

        st.info("ADIM 2: 5x5 Grid analizi başlıyor...")
        safe_np = np.array(image.convert('RGB')) if isinstance(image, Image.Image) else image
        grid_analysis = self.analyze_5x5_grid_original(
            safe_np,
            spacing_mm=spacing_mm,
            strict_cell_damage=strict_cell_damage,
            flake_ratio_thr=flake_ratio_thr,
            min_pix_ratio=min_pix_ratio,
            return_debug=return_debug
        )
        st.success("Grid analizi tamamlandı!")

        st.info("ADIM 3: Model tahmini başlıyor...")
        if not self.using_demo_model:
            # Eğitimli model varsa onu kullan
            predictions = self.model.predict(processed_input, verbose=0)[0]
        else:
            # Kural tabanlı olasılık: güven sabit değil, hasar şiddetine göre
            predictions = self.generate_rule_based_predictions(
                damaged_cells_count=round(grid_analysis['damaged_cells'])
            )
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])
        st.success("Model tahmini tamamlandı!")

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': predictions.tolist(),
            'grid_analysis': grid_analysis,
            'preprocessing_steps': preprocessing_steps,
            'class_info': self.iso_classes[predicted_class],
            'used_demo_model': self.using_demo_model
        }

    def generate_rule_based_predictions(self, damaged_cells_count: int):
        """
        Sınıf eşleme tamamen hücre adedine göre:
          0 hücre -> Class 0
          1 hücre -> Class 1
          2-3     -> Class 2
          4-8     -> Class 3
          9-16    -> Class 4
          17-25   -> Class 5
        Güven, hasar yüzdesine göre 0.55–0.95 arası ölçeklenir.
        """
        n = int(damaged_cells_count)
        if n == 0:
            dominant = 0
        elif n == 1:
            dominant = 1
        elif 2 <= n <= 3:
            dominant = 2
        elif 4 <= n <= 8:
            dominant = 3
        elif 9 <= n <= 16:
            dominant = 4
        else:
            dominant = 5

        severity = np.clip(n / 25.0, 0.0, 1.0)
        main_p = 0.55 + 0.40 * severity
        side_p = (1.0 - main_p) / 2.0

        probs = np.zeros(6, dtype=np.float32)
        probs[dominant] = main_p
        if dominant - 1 >= 0: probs[dominant - 1] = side_p
        if dominant + 1 <= 5: probs[dominant + 1] = side_p
        return probs / probs.sum()

# ------------------------------------------------------------
# Uygulama ana akışı
# ------------------------------------------------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ISO 2409 Çapraz Kesim Test Sınıflandırıcısı</h1>
        <p>Kesik çizgilerini maskeleyerek yalnızca hücre içi kopmaları değerlendirir</p>
    </div>
    """, unsafe_allow_html=True)

    if 'classifier' not in st.session_state:
        st.session_state.classifier = CrosscutClassifier()
    classifier = st.session_state.classifier

    # Sidebar: standart ve eşikler
    with st.sidebar:
        st.header("📋 ISO 2409:2013")
        st.info("📌 Kural: **Hücre içinde kopma yoksa = Class 0**. En küçük kopma varsa sınıf ≥1.")
        spacing_mm = st.radio("Kesik aralığı (mm)", [1, 2, 3], index=0, horizontal=True)
        strict_mode = st.checkbox("Katı hücre kuralı (tavsiye)", value=True)
        st.markdown("---")
        st.subheader("Eşikler (ince ayar)")
        flake_ratio_thr = st.slider("Hücre hasar oran eşiği", 0.001, 0.02, 0.003, 0.001, help="Hücre içindeki flake oranı eşiği")
        min_pix_ratio   = st.slider("Hücre başı min piksel (oran)", 0.0005, 0.01, 0.001, 0.0005, help="Hücre alanına oranla min flake pikseli")
        show_debug_masks = st.checkbox("Maske debug görüntülerini göster", value=False)
        st.markdown("---")
        st.write("Sınıf renkleri ve açıklamalar:")
        for i, class_info in classifier.iso_classes.items():
            st.markdown(f"- **Sınıf {i} – {class_info['quality']}**: {class_info['description']}")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.header("Görüntü Yükleme")
        uploaded_file = st.file_uploader("Çapraz kesim test görüntüsünü yükleyin", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görüntü")

            # --- Otomatik + Elle ayarlanabilir kare crop ---
            st.subheader("Kare Seçimi (Otomatik + Elle Ayar)")
            img_width, img_height = image.size
            try:
                auto_region = classifier.detect_crosscut_grid_region(image)
                auto_size = int(min(auto_region['width'], auto_region['height']))
                auto_cx  = int(auto_region['x'] + auto_region['width'] // 2)
                auto_cy  = int(auto_region['y'] + auto_region['height'] // 2)
            except Exception:
                auto_cx, auto_cy = img_width // 2, img_height // 2
                auto_size = int(min(img_width, img_height) * 0.6)

            col_sq1, col_sq2, col_sq3 = st.columns(3)
            with col_sq1:
                cx = st.slider("Merkez X", 0, img_width - 1, auto_cx, step=1)
            with col_sq2:
                cy = st.slider("Merkez Y", 0, img_height - 1, auto_cy, step=1)
            with col_sq3:
                size = st.slider("Kare Boyutu", 50, min(img_width, img_height), auto_size, step=5)

            preview = draw_square_overlay(image, cx, cy, size, color=(0,255,0), thickness=3)
            st.image(preview, caption="Ön İzleme (Kare konturlu)")

            if st.button("Kırp ve Analiz Et", type="primary"):
                cropped_image = crop_square(image, cx, cy, size)
                st.image(cropped_image, caption=f"Kırpılmış Görüntü ({size}x{size})")

                st.subheader("Kontrast Ayarlama")
                contrast = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1, key="contrast_after_crop")
                brightness = st.slider("Parlaklık", 0.5, 2.0, 1.0, 0.1, key="brightness_after_crop")

                final_image = cropped_image
                if contrast != 1.0 or brightness != 1.0:
                    enhancer = ImageEnhance.Contrast(cropped_image); adj_image = enhancer.enhance(contrast)
                    enhancer = ImageEnhance.Brightness(adj_image);  adj_image = enhancer.enhance(brightness)
                    st.image(adj_image, caption=f"Ayarlanmış (K:{contrast:.1f}, P:{brightness:.1f})")
                    final_image = adj_image

                result = classifier.predict(
                    final_image,
                    spacing_mm=int(spacing_mm),
                    strict_cell_damage=bool(strict_mode),
                    flake_ratio_thr=float(flake_ratio_thr),
                    min_pix_ratio=float(min_pix_ratio),
                    return_debug=bool(show_debug_masks)
                )
                if result:
                    st.session_state.prediction_result = result
            else:
                st.info("▶ Kareyi konumlandırıp **Kırp ve Analiz Et** butonuna basın.")

    with col2:
        st.header("📊 Analiz Sonuçları")
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            class_info = result['class_info']
            g = result['grid_analysis']

            st.markdown(f"""
            <div class="prediction-box class-{result['predicted_class']}" style="border-color: {class_info['color']}">
                <h2>{class_info['name']}</h2>
                <h3>{class_info['quality']}</h3>
                <p><strong>{class_info['description']}</strong></p>
                <p>{class_info['criteria']}</p>
            </div>
            """, unsafe_allow_html=True)

            col_conf, col_delam, col_cells = st.columns(3)
            with col_conf:
                st.metric("Güven Seviyesi", f"{result['confidence']:.1%}")
            with col_delam:
                st.metric("Ayrılma Oranı", f"{g['delamination_ratio']:.2f}%")
            with col_cells:
                st.metric("Hasarlı Hücre Adedi", f"{round(g['damaged_cells'])}/25")

            st.subheader("🎯 Grid Analiz Sonuçları")
            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                st.metric("Grid Kalite Skoru", f"{g['grid_quality_score']:.1f}/100")
                st.metric("Hasar Yüzdesi", f"{g['damage_percentage']:.1f}%")
            with grid_col2:
                grid_status = "✅ Tespit Edildi" if g['grid_detected'] else "❌ Tespit Edilemedi"
                st.info(f"**5x5 Grid Durumu:** {grid_status}\n\nKesik kalınlığı (px): {g['cut_thickness_px']}, px/mm: {g['px_per_mm']:.1f}")

            if result.get('used_demo_model', True):
                st.caption("ℹ️ Eğitimli model bulunamadı; kural tabanlı tahmin kullanıldı.")

            st.subheader("📈 Sınıf Olasılıkları")
            prob_data = pd.DataFrame({
                'Sınıf': [f"Sınıf {i}" for i in range(6)],
                'Olasılık': [prob * 100 for prob in result['probabilities']],
                'Renk': [classifier.iso_classes[i]['color'] for i in range(6)]
            })
            fig = px.bar(prob_data, x='Sınıf', y='Olasılık',
                         color='Renk', color_discrete_map={c: c for c in prob_data['Renk']},
                         title="ISO Sınıf Olasılık Dağılımı")
            fig.update_layout(showlegend=False, height=400, yaxis_title="Olasılık (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Analiz için kareyi seçip kırpın")

    if 'prediction_result' in st.session_state:
        st.markdown("---")
        with st.expander("🔬 Detaylı Grid Analizi"):
            g = st.session_state.prediction_result['grid_analysis']
            st.json({
                "Analiz Metodu": g.get('analysis_method'),
                "Yüzey Tipi": g.get('surface_type'),
                "Grid Bölgesi": f"x:{g['grid_region']['x']}, y:{g['grid_region']['y']}, w:{g['grid_region']['width']}, h:{g['grid_region']['height']}",
                "Kesik Kalınlığı (px)": g.get('cut_thickness_px'),
                "px/mm": f"{g.get('px_per_mm', 0):.2f}",
                "Hücre Alanı (px^2)": g.get('cell_area_px'),
                "Hücre Eşiği (oran)": g.get('flake_ratio_thr'),
                "Hücre Eşiği (min px)": g.get('min_pix'),
                "Hasarlı Hücre Adedi": f"{round(g['damaged_cells'])}/25",
                "Hasar Yüzdesi": f"{g['damage_percentage']:.1f}%",
                "Ayrılma Oranı (kesik hariç)": f"{g['delamination_ratio']:.2f}%"
            })
            # Debug maskeleri göster
            if 'debug_masks' in g:
                st.subheader("🧪 Maske Debug")
                dm = g['debug_masks']
                st.image(dm['cut_mask'], caption="Kesik Maskesi", clamp=True)
                st.image(dm['flake_mask'], caption="Flake Maskesi (adaptif)", clamp=True)
                st.image(dm['flake_mask_interior'], caption="Kesik Hariç Flake", clamp=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.info("🎯 **Özellik:** 5x5 grid otomatik tespit")
    with c2: st.info("🧠 **Maske:** Kesik çizgileri hariç tutulur")
    with c3: st.info("🔧 **Kural:** Hücre içi en ufak kopma ⇒ sınıf ≥ 1")

# ------------------------------------------------------------
# Çalıştırma
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
