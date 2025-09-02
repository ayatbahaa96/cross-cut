# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px
import pandas as pd
import json
from pathlib import Path

st.set_page_config(
    page_title="ISO 2409 Çapraz Kesim Sınıflandırıcısı",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
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
    .class-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .class-0 { border-color: #27ae60; }
    .class-1 { border-color: #2ecc71; }
    .class-2 { border-color: #f1c40f; }
    .class-3 { border-color: #e67e22; }
    .class-4 { border-color: #e74c3c; }
    .class-5 { border-color: #c0392b; }
    .prediction-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CrosscutClassifier:
    def __init__(self, model_path="models/crosscut_model.keras", labels_path="models/class_indices.json"):
        self.iso_classes = {
            0: {"name": "Sınıf 0", "description": "Kesilerin kesişim noktalarında bozulma yok",
                "criteria": "Mükemmel yapışma, hiç ayrılma yok", "quality": "Mükemmel", "color": "#27ae60"},
            1: {"name": "Sınıf 1", "description": "Kesişim noktalarında çok küçük pullar",
                "criteria": "Sadece kesişim noktalarında minimal ayrılma", "quality": "Çok İyi", "color": "#2ecc71"},
            2: {"name": "Sınıf 2", "description": "Kesişim noktalarında ve/veya kesiler boyunca küçük pullar",
                "criteria": "Kesim kenarları boyunca küçük ayrılmalar", "quality": "İyi", "color": "#f1c40f"},
            3: {"name": "Sınıf 3", "description": "Büyük pullar ve/veya kesim kenarları boyunca büyük pullar",
                "criteria": "Karelere doğru uzanan büyük ayrılmalar", "quality": "Kabul Edilebilir", "color": "#e67e22"},
            4: {"name": "Sınıf 4", "description": "Büyük pullar. Çapraz kesim alanının %5'den fazla kısmı bozulur",
                "criteria": "Önemli alan etkilenmiş, belirgin ayrılma", "quality": "Zayıf", "color": "#e74c3c"},
            5: {"name": "Sınıf 5", "description": "Herhangi bir derece, sıkıntılanma pulları",
                "criteria": "Çok zayıf yapışma, yaygın ayrılma", "quality": "Çok Zayıf", "color": "#c0392b"}
        }
        self.model = None
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.load_model()

    def load_model(self):
        try:
            if not self.model_path.exists():
                st.warning("⚠️ Model dosyası bulunamadı. Lütfen önce `train_crosscut.py` ile modeli eğitin.")
                return False
            self.model = tf.keras.models.load_model(self.model_path)
            # (opsiyonel) etiket eşleşmesi için dosya okunabilir
            if self.labels_path.exists():
                with open(self.labels_path, "r") as f:
                    self.class_indices = json.load(f)
            return True
        except Exception as e:
            st.error(f"Model yüklenemedi: {e}")
            return False

    def preprocess_image(self, image):
        # PIL -> numpy, RGB
        if isinstance(image, Image.Image):
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

        # ensure RGB
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, (224, 224))
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Görüntü RGB formatında değil. Şekil: {image.shape}")

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def analyze_grid_pattern(self, image):
        # PIL -> np
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

        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            edges = cv2.Canny(enhanced, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)

            dark_pixels = np.sum(gray < 100)
            total_pixels = gray.shape[0] * gray.shape[1]
            delamination_ratio = (dark_pixels / total_pixels) * 100
            edge_density = np.sum(edges > 0) / total_pixels * 100

            return {
                'delamination_ratio': float(delamination_ratio),
                'edge_density': float(edge_density),
                'grid_quality': 'İyi' if lines is not None and len(lines) > 10 else 'Zayıf'
            }
        except Exception:
            return {
                'delamination_ratio': 10.0,
                'edge_density': 5.0,
                'grid_quality': 'Belirlenemedi'
            }

    def predict(self, image):
        if self.model is None:
            return None
        processed = self.preprocess_image(image)
        grid = self.analyze_grid_pattern(image)
        preds = self.model.predict(processed, verbose=0)[0]
        predicted_class = int(np.argmax(preds))
        confidence = float(preds[predicted_class])
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': preds.tolist(),
            'grid_analysis': grid,
            'class_info': self.iso_classes[predicted_class]
        }

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ISO 2409 Çapraz Kesim Test Sınıflandırıcısı</h1>
        <p>Yapışma dayanımı otomatik değerlendirme sistemi</p>
    </div>
    """, unsafe_allow_html=True)

    if 'classifier' not in st.session_state:
        st.session_state.classifier = CrosscutClassifier()
    classifier = st.session_state.classifier

    with st.sidebar:
        st.header("📋 ISO 2409:2013 Standartı")
        for i, class_info in classifier.iso_classes.items():
            with st.expander(f"Sınıf {i} - {class_info['quality']}"):
                st.write(f"**Tanım:** {class_info['description']}")
                st.write(f"**Kriter:** {class_info['criteria']}")
        st.markdown("---")
        st.info("📌 **Not:** Görüntü yükledikten sonra analiz edebilirsiniz.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.header("📤 Görüntü Yükleme")
        uploaded_file = st.file_uploader(
            "Çapraz kesim test görüntüsünü yükleyin",
            type=['png', 'jpg', 'jpeg'],
            help="JPG, PNG formatlarında görüntü yükleyebilirsiniz"
        )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

            if st.button("🔍 Analiz Et", type="primary", disabled=classifier.model is None):
                with st.spinner("Çapraz kesim deseni analiz ediliyor..."):
                    result = classifier.predict(image)
                    if result:
                        st.session_state.prediction_result = result
                        st.success("Analiz tamamlandı!")

        if uploaded_file is None and classifier.model is None:
            st.info("👆 Başlamak için modeli eğitin ve görüntü yükleyin.")

    with col2:
        st.header("📊 Analiz Sonuçları")
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            class_info = result['class_info']
            st.markdown(f"""
            <div class="prediction-box class-{result['predicted_class']}" style="border-color: {class_info['color']}">
                <h2>{class_info['name']}</h2>
                <h3>{class_info['quality']}</h3>
                <p><strong>{class_info['description']}</strong></p>
                <p>{class_info['criteria']}</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Güven Seviyesi", f"{result['confidence']:.1%}", help="Tahminin güvenilirlik seviyesi")
            with c2:
                st.metric("Tahmini Ayrılma", f"{result['grid_analysis']['delamination_ratio']:.1f}%",
                          help="Grid analizi sonucu ayrılma oranı")

            st.subheader("📈 Sınıf Olasılıkları")
            prob_data = pd.DataFrame({
                'Sınıf': [f"Sınıf {i}" for i in range(6)],
                'Olasılık': [p * 100 for p in result['probabilities']],
                'Renk': [classifier.iso_classes[i]['color'] for i in range(6)]
            })
            fig = px.bar(
                prob_data, x='Sınıf', y='Olasılık',
                color='Renk',
                color_discrete_map={c: c for c in prob_data['Renk']},
                title="ISO Sınıf Olasılık Dağılımı"
            )
            fig.update_layout(showlegend=False, height=400, yaxis_title="Olasılık (%)")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔬 Detaylı Analiz Sonuçları"):
                st.json({
                    "Tahmin Edilen Sınıf": result['predicted_class'],
                    "Güven Seviyesi": f"{result['confidence']:.3f}",
                    "Grid Kalitesi": result['grid_analysis']['grid_quality'],
                    "Kenar Yoğunluğu": f"{result['grid_analysis']['edge_density']:.1f}%",
                    "Ayrılma Oranı": f"{result['grid_analysis']['delamination_ratio']:.1f}%"
                })
        else:
            st.info("👆 Analiz için görüntü yükleyin")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🎯 **Doğruluk:** Eğitim sonrası %90+ hedef")
    with c2:
        st.info("⚡ **Hız:** Görüntü başına ~2 sn")
    with c3:
        st.info("🔧 **Model:** TensorFlow/Keras + EfficientNetB0")

if __name__ == "__main__":
    main()
