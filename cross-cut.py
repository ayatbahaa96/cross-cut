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
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f8f9fa; padding: 2rem; border-radius: 15px; border: 3px solid; text-align: center; margin: 1rem 0;
    }
    .preprocessing-steps {
        background: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 1rem 0;
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

# --- Kesik çizgisi maskesi
def make_cut_mask(grid_gray: np.ndarray, thickness_px: int) -> np.ndarray:
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

# ------------------------------------------------------------
# Ön işleme adımları gösterimi
# ------------------------------------------------------------
def show_preprocessing_steps(steps):
    st.markdown("""
    <div class="preprocessing-steps">
        <h3>🔄 Görüntü Ön İşleme Adımları</h3>
        <p>5x5 grid yapısı için optimize edilmiş işleme pipeline'ı</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### İlk Satır")
    c1, c2, c3, c4 = st.columns(4)
    if 'original' in steps:
        with c1: st.image(cv2.resize(steps['original'], (250,250)), caption="Orijinal")
    if 'grayscale' in steps:
        with c2: st.image(cv2.resize(cv2.cvtColor(steps['grayscale'], cv2.COLOR_GRAY2RGB),(250,250)), caption="Gri")
    if 'denoised' in steps:
        with c3: st.image(cv2.resize(cv2.cvtColor(steps['denoised'], cv2.COLOR_GRAY2RGB),(250,250)), caption="Denoised")
    if 'contrast_enhanced' in steps:
        with c4: st.image(cv2.resize(cv2.cvtColor(steps['contrast_enhanced'], cv2.COLOR_GRAY2RGB),(250,250)), caption="CLAHE")
    st.markdown("### İkinci Satır")
    c1, c2, c3, c4, c5 = st.columns(5)
    if 'bilateral' in steps:
        with c1: st.image(cv2.resize(cv2.cvtColor(steps['bilateral'], cv2.COLOR_GRAY2RGB),(200,200)), caption="Bilateral")
    if 'morphology' in steps:
        with c2: st.image(cv2.resize(cv2.cvtColor(steps['morphology'], cv2.COLOR_GRAY2RGB),(200,200)), caption="Morfoloji")
    if 'adaptive_threshold' in steps:
        with c3: st.image(cv2.resize(cv2.cvtColor(steps['adaptive_threshold'], cv2.COLOR_GRAY2RGB),(200,200)), caption="Adaptif Eşik")
    if 'grid_lines' in steps:
        with c4: st.image(cv2.resize(cv2.cvtColor(steps['grid_lines'], cv2.COLOR_GRAY2RGB),(200,200)), caption="Grid Çizgileri")
    if 'final_processed' in steps:
        with c5: st.image(cv2.resize(cv2.cvtColor(steps['final_processed'], cv2.COLOR_GRAY2RGB),(200,200)), caption="Final")

# ------------------------------------------------------------
# Sınıflandırıcı
# ------------------------------------------------------------
class CrosscutClassifier:
    def __init__(self):
        self.iso_classes = {
            0: {"name": "Sınıf 0","description": "Kopma yok","criteria": "Mükemmel yapışma","quality": "Mükemmel","color": "#27ae60"},
            1: {"name": "Sınıf 1","description": "Çok küçük pullar","criteria": "Kesişim nok./kenarlarda minimal","quality": "Çok İyi","color": "#2ecc71"},
            2: {"name": "Sınıf 2","description": "Küçük pullar","criteria": "Kesim kenarları boyunca küçük","quality": "İyi","color": "#f1c40f"},
            3: {"name": "Sınıf 3","description": "Belirgin pullar","criteria": "Karelere ilerleyen","quality": "Kabul","color": "#e67e22"},
            4: {"name": "Sınıf 4","description": "Geniş pullar (>~%5)","criteria": "Önemli alan etkilenmiş","quality": "Zayıf","color": "#e74c3c"},
            5: {"name": "Sınıf 5","description": "Yaygın ayrılma","criteria": "Çok zayıf yapışma","quality": "Çok Zayıf","color": "#c0392b"}
        }
        self.model = None
        self.using_demo_model = True
        self.load_model()

    def load_model(self):
        try:
            model_path = "model_iso2409.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.using_demo_model = False
            else:
                self.model = self.create_demo_model()
                self.using_demo_model = True
        except Exception:
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

        steps = {}
        original = image_array.copy()
        h, w = original.shape[:2]
        if h != w:
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            original = cv2.copyMakeBorder(original, pad_h, pad_h, pad_w, pad_w,
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
        original = cv2.resize(original, (400, 400))
        steps['original'] = original

        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY); steps['grayscale'] = gray
        denoised = cv2.fastNlMeansDenoising(gray); steps['denoised'] = denoised
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised); steps['contrast_enhanced'] = contrast_enhanced
        bilateral = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75); steps['bilateral'] = bilateral
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(bilateral, cv2.MORPH_OPEN, kernel); steps['morphology'] = opening
        adaptive_thresh = cv2.adaptiveThreshold(
            opening, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ); steps['adaptive_threshold'] = adaptive_thresh
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel)
        grid_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0); steps['grid_lines'] = grid_lines
        final_processed = cv2.addWeighted(opening, 0.8, grid_lines, 0.2, 0); steps['final_processed'] = final_processed

        processed_rgb = cv2.cvtColor(final_processed, cv2.COLOR_GRAY2RGB)
        processed_rgb = cv2.resize(processed_rgb, (224, 224)).astype(np.float32) / 255.0
        model_input = np.expand_dims(processed_rgb, axis=0)
        return model_input, steps

    # ------------------------- GRID BÖLGESİ -------------------------
    def detect_crosscut_grid_region(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
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

    # ------------------------- 5x5 Analizi (Adaptif eşikle karar) -------------------------
    def analyze_5x5_grid_original(
        self,
        original_image,
        spacing_mm: int = 1,
        use_adaptive_as_flake: bool = True,
        invert_adaptive: bool = False,
        cell_ratio_thr: float = 0.003,
        min_pix_ratio: float = 0.001,
        return_debug: bool = False
    ):
        # Görseli hazırla
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

        gray_full = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY) if work_image.ndim == 3 else work_image
        region = self.detect_crosscut_grid_region(work_image)
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        grid_gray = gray_full[y:y+h, x:x+w]
        grid_rgb  = work_image[y:y+h, x:x+w] if work_image.ndim == 3 else cv2.cvtColor(grid_gray, cv2.COLOR_GRAY2RGB)

        # Grid kalite skoru (yaklaşık)
        edges = cv2.Canny(grid_gray, 30, 100)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        grid_quality_score = 0
        if lines is not None:
            horiz, vert = [], []
            for line in lines:
                rho, theta = line[0]
                if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4: horiz.append(line)
                elif abs(theta - np.pi/2) < np.pi/4: vert.append(line)
            grid_quality_score = min(len(horiz), len(vert)) / 6 * 100

        # Hücre boyutu / px/mm
        height, width = grid_gray.shape
        cell_h = max(1, height // 5); cell_w = max(1, width // 5)
        px_per_mm = max(1.0, ((cell_w + cell_h) / 2.0) / float(spacing_mm))
        thickness_px = int(np.clip(round(0.35 * px_per_mm), 2, 14))
        cell_area = cell_w * cell_h
        MIN_PIX = max(5, int(min_pix_ratio * cell_area))
        CELL_RATIO_THR = float(cell_ratio_thr)

        # 1) Kesik maskesi
        cut_mask = make_cut_mask(grid_gray, thickness_px)
        interior_mask = cv2.bitwise_not(cut_mask)

        # 2) Flake maskesi = Adaptif eşik (siyah = zarar)
        if use_adaptive_as_flake:
            mode = cv2.THRESH_BINARY_INV if invert_adaptive else cv2.THRESH_BINARY
            adaptive = cv2.adaptiveThreshold(
                grid_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 11, 2
            )
            # hasar = siyah (0) pikseller olsun -> invert et
            flake_mask = (adaptive == 0).astype(np.uint8) * 255
            flake_mask = cv2.morphologyEx(flake_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        else:
            # Yedek: basit global Otsu (koyu = zarar)
            _, otsu = cv2.threshold(grid_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            flake_mask = (otsu == 0).astype(np.uint8) * 255

        # 3) Kesik hariç hasar
        flake_mask_interior = cv2.bitwise_and(flake_mask, interior_mask)

        # Hücre bazında ölçüm: fraksiyonel katkı
        damaged_cells_binary = 0.0
        damaged_cells_eq = 0.0
        cell_ratios = []
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
                cell_ratios.append(float(ratio))

                # binary hasarlı hücre sayısı
                if flake_pix >= MIN_PIX and ratio >= CELL_RATIO_THR:
                    damaged_cells_binary += 1.0

                # fraksiyonel hücre katkısı (0..1)
                contrib = 0.0
                if flake_pix >= MIN_PIX:
                    contrib = min(1.0, ratio / CELL_RATIO_THR)
                damaged_cells_eq += contrib

        delamination_ratio = 100.0 * (flake_interior_total / interior_total) if interior_total > 0 else 0.0

        out = {
            'grid_quality_score': float(grid_quality_score),
            'delamination_ratio': float(delamination_ratio),
            'damaged_cells_binary': float(damaged_cells_binary),
            'damaged_cells_eq': float(damaged_cells_eq),
            'cell_ratios': cell_ratios,
            'total_cells': 25,
            'damage_percentage_binary': (damaged_cells_binary / 25.0) * 100.0,
            'grid_detected': bool(lines is not None and len(lines) > 8),
            'grid_region': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'spacing_mm': spacing_mm,
            'px_per_mm': px_per_mm,
            'cut_thickness_px': thickness_px,
            'cell_area_px': cell_area,
            'cell_ratio_thr': CELL_RATIO_THR,
            'min_pix': MIN_PIX,
            'use_adaptive_as_flake': bool(use_adaptive_as_flake),
            'invert_adaptive': bool(invert_adaptive)
        }
        if return_debug:
            out['debug_masks'] = {
                'cut_mask': cut_mask,
                'adaptive_binary': adaptive if use_adaptive_as_flake else None,
                'flake_mask': flake_mask,
                'flake_mask_interior': flake_mask_interior
            }
        return out

    # ------------------------- Kurala göre sınıf belirleme -------------------------
    @staticmethod
    def map_cells_to_class(n_eq: float) -> int:
        if n_eq <= 0.0:
            return 0                     # hiç zarar yoksa Class 0
        elif n_eq <= 1.25:
            return 1
        elif n_eq <= 3.5:
            return 2
        elif n_eq <= 8.5:
            return 3
        elif n_eq <= 16.5:
            return 4
        else:
            return 5

    def rule_based_probabilities(self, n_eq: float) -> np.ndarray:
        dom = self.map_cells_to_class(n_eq)
        # güveni hasar seviyesine göre 0.60–0.95 arası ölçekle
        severity = np.clip(n_eq / 25.0, 0.0, 1.0)
        main_p = 0.60 + 0.35 * severity
        side_p = (1.0 - main_p) / 2.0
        probs = np.zeros(6, dtype=np.float32)
        probs[dom] = main_p
        if dom - 1 >= 0: probs[dom - 1] = side_p
        if dom + 1 <= 5: probs[dom + 1] = side_p
        return probs / probs.sum()

    # ------------------------- PREDICT -------------------------
    def predict(
        self,
        image,
        spacing_mm: int = 1,
        decide_with_adaptive: bool = True,
        invert_adaptive: bool = False,
        cell_ratio_thr: float = 0.003,
        min_pix_ratio: float = 0.001,
        return_debug: bool = False
    ):
        st.info("ADIM 1: Ön işleme…")
        processed_input, steps = self.enhanced_preprocessing(image)
        st.success("Ön işleme tamam.")
        show_preprocessing_steps(steps)

        st.info("ADIM 2: 5x5 grid analizi…")
        safe_np = np.array(image.convert('RGB')) if isinstance(image, Image.Image) else image
        g = self.analyze_5x5_grid_original(
            safe_np,
            spacing_mm=spacing_mm,
            use_adaptive_as_flake=decide_with_adaptive,
            invert_adaptive=invert_adaptive,
            cell_ratio_thr=cell_ratio_thr,
            min_pix_ratio=min_pix_ratio,
            return_debug=return_debug
        )
        st.success("Grid analizi tamam.")

        st.info("ADIM 3: Sınıf tahmini…")
        if decide_with_adaptive:
            n_eq = g['damaged_cells_eq']
            probs = self.rule_based_probabilities(n_eq)
            pred_cls = int(np.argmax(probs))
            conf = float(probs[pred_cls])
            decision_src = "adaptive_threshold_rule"
        else:
            if self.using_demo_model:
                # eğitimli model yoksa yine kurala dön
                n_eq = g['damaged_cells_eq']
                probs = self.rule_based_probabilities(n_eq)
                pred_cls = int(np.argmax(probs))
                conf = float(probs[pred_cls])
                decision_src = "adaptive_threshold_rule (no trained model)"
            else:
                probs = self.model.predict(processed_input, verbose=0)[0]
                pred_cls = int(np.argmax(probs))
                conf = float(probs[pred_cls])
                decision_src = "trained_model"

        return {
            'predicted_class': pred_cls,
            'confidence': conf,
            'probabilities': probs.tolist(),
            'grid_analysis': g,
            'preprocessing_steps': steps,
            'class_info': self.iso_classes[pred_cls],
            'decision_source': decision_src
        }

# ------------------------------------------------------------
# Uygulama
# ------------------------------------------------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ISO 2409 Çapraz Kesim Test Sınıflandırıcısı</h1>
        <p>Adaptif eşik görüntüsünden fraksiyonel hücre zararı ile sınıf belirleme</p>
    </div>
    """, unsafe_allow_html=True)

    if 'classifier' not in st.session_state:
        st.session_state.classifier = CrosscutClassifier()
    clf = st.session_state.classifier

    # Sidebar
    with st.sidebar:
        st.header("📋 ISO 2409:2013")
        st.info("📌 En ufak kopma > 0 ise Class ≥ 1. (0 → Class 0)")
        spacing_mm = st.radio("Kesik aralığı (mm)", [1, 2, 3], index=0, horizontal=True)
        decide_with_adaptive = st.checkbox("Sınıfı adaptif eşik görüntüsüne göre belirle", value=True)
        invert_adaptive = st.checkbox("Adaptif eşiği tersle (beyaz=zarar)", value=False)
        cell_ratio_thr = st.slider("Hücre oran eşiği", 0.001, 0.02, 0.003, 0.001,
                                   help="Bir hücrenin 'tam hasarlı' sayılması için gereken oran")
        min_pix_ratio = st.slider("Min piksel (hücre oranı)", 0.0005, 0.01, 0.001, 0.0005,
                                  help="Gürültüyü elemek için minimum hasar pikseli")
        show_debug = st.checkbox("Maske / adaptif görüntüleri göster", value=False)
        st.markdown("---")
        st.write("Sınıf açıklamaları:")
        for i, info in clf.iso_classes.items():
            st.markdown(f"- **Sınıf {i} – {info['quality']}**: {info['description']}")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.header("Görüntü Yükleme")
        uploaded_file = st.file_uploader("Test görüntüsü yükleyin", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görüntü")

            # Otomatik + elle kare seçim
            st.subheader("Kare Seçimi (Otomatik + Elle)")
            w, h = image.size
            try:
                auto = clf.detect_crosscut_grid_region(image)
                auto_size = int(min(auto['width'], auto['height']))
                auto_cx  = int(auto['x'] + auto['width'] // 2)
                auto_cy  = int(auto['y'] + auto['height'] // 2)
            except Exception:
                auto_cx, auto_cy = w // 2, h // 2
                auto_size = int(min(w, h) * 0.6)

            c1, c2, c3 = st.columns(3)
            with c1: cx = st.slider("Merkez X", 0, w - 1, auto_cx, step=1)
            with c2: cy = st.slider("Merkez Y", 0, h - 1, auto_cy, step=1)
            with c3: size = st.slider("Kare Boyutu", 50, min(w, h), auto_size, step=5)

            preview = draw_square_overlay(image, cx, cy, size)
            st.image(preview, caption="Ön İzleme")

            if st.button("Kırp ve Analiz Et", type="primary"):
                cropped = crop_square(image, cx, cy, size)
                st.image(cropped, caption=f"Kırpılmış ({size}x{size})")

                st.subheader("Kontrast Ayarı")
                contrast = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1, key="ct")
                brightness = st.slider("Parlaklık", 0.5, 2.0, 1.0, 0.1, key="br")

                final_image = cropped
                if contrast != 1.0 or brightness != 1.0:
                    e = ImageEnhance.Contrast(cropped); img2 = e.enhance(contrast)
                    e = ImageEnhance.Brightness(img2);  img2 = e.enhance(brightness)
                    st.image(img2, caption=f"Ayarlanmış (K:{contrast:.1f}, P:{brightness:.1f})")
                    final_image = img2

                result = clf.predict(
                    final_image,
                    spacing_mm=int(spacing_mm),
                    decide_with_adaptive=bool(decide_with_adaptive),
                    invert_adaptive=bool(invert_adaptive),
                    cell_ratio_thr=float(cell_ratio_thr),
                    min_pix_ratio=float(min_pix_ratio),
                    return_debug=bool(show_debug)
                )
                if result:
                    st.session_state.prediction_result = result
            else:
                st.info("▶ Kareyi konumlandırıp **Kırp ve Analiz Et** butonuna basın.")

    with col2:
        st.header("📊 Analiz Sonuçları")
        if 'prediction_result' in st.session_state:
            r = st.session_state.prediction_result
            info = r['class_info']; g = r['grid_analysis']

            st.markdown(f"""
            <div class="prediction-box class-{r['predicted_class']}" style="border-color: {info['color']}">
                <h2>{info['name']}</h2>
                <h3>{info['quality']}</h3>
                <p><strong>{info['description']}</strong></p>
                <p>{info['criteria']}</p>
            </div>
            """, unsafe_allow_html=True)

            colA, colB, colC = st.columns(3)
            with colA: st.metric("Güven", f"{r['confidence']:.1%}")
            with colB: st.metric("Ayrılma Oranı", f"{g['delamination_ratio']:.2f}%")
            with colC: st.metric("Hasarlı Hücre (eşdeğer)", f"{g['damaged_cells_eq']:.2f}/25")

            st.subheader("🎯 Grid Analiz Özeti")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Grid Skoru", f"{g['grid_quality_score']:.1f}/100")
                st.metric("Binary Hasar Yüzdesi", f"{g['damage_percentage_binary']:.1f}%")
            with c2:
                grid_status = "✅ Tespit Edildi" if g['grid_detected'] else "❌ Tespit Edilemedi"
                st.info(f"**5x5 Grid Durumu:** {grid_status}\n\nKesik kalınlığı (px): {g['cut_thickness_px']}, px/mm: {g['px_per_mm']:.1f}")
                st.caption(f"Karar kaynağı: {r['decision_source']}")

            st.subheader("📈 Sınıf Olasılıkları")
            df = pd.DataFrame({
                'Sınıf': [f"Sınıf {i}" for i in range(6)],
                'Olasılık': [p * 100 for p in r['probabilities']],
                'Renk': [clf.iso_classes[i]['color'] for i in range(6)]
            })
            fig = px.bar(df, x='Sınıf', y='Olasılık',
                         color='Renk', color_discrete_map={c: c for c in df['Renk']},
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
                "Grid Bölgesi": f"x:{g['grid_region']['x']}, y:{g['grid_region']['y']}, w:{g['grid_region']['width']}, h:{g['grid_region']['height']}",
                "Kesik Kalınlığı (px)": g['cut_thickness_px'],
                "px/mm": f"{g['px_per_mm']:.2f}",
                "Hücre Alanı (px^2)": g['cell_area_px'],
                "Hücre Oran Eşiği": g['cell_ratio_thr'],
                "Min Piksel": g['min_pix'],
                "Binary Hasarlı Hücre": g['damaged_cells_binary'],
                "Eşdeğer Hasarlı Hücre": round(g['damaged_cells_eq'], 3),
                "Ayrılma Oranı": f"{g['delamination_ratio']:.2f}%"
            })
            if 'debug_masks' in g:
                st.subheader("🧪 Debug Görselleri")
                dm = g['debug_masks']
                if dm.get('adaptive_binary') is not None:
                    st.image(dm['adaptive_binary'], caption="Adaptif İkili (beyaz=zemin, siyah=zarar?)", clamp=True)
                st.image(dm['cut_mask'], caption="Kesik Maskesi", clamp=True)
                st.image(dm['flake_mask'], caption="Hasar Maskesi", clamp=True)
                st.image(dm['flake_mask_interior'], caption="Kesik Hariç Hasar", clamp=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.info("🎯 **Karar:** Adaptif eşik + fraksiyonel hücre")
    with c2: st.info("🧠 **Maske:** Kesikler hariç")
    with c3: st.info("🔧 **Kural:** 0→C0, (0–1.25]→C1, …")

# ------------------------------------------------------------
# Çalıştırma
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
