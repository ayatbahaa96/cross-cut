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
st.set_page_config(page_title="ISO 2409 Çapraz Kesim Sınıflandırıcısı", page_icon="🔬", layout="wide")

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
 .main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center;margin-bottom:1rem}
 .prediction-box{background:#f8f9fa;padding:1.2rem;border-radius:15px;border:3px solid;text-align:center;margin:1rem 0}
 .class-0{border-color:#27ae60}.class-1{border-color:#2ecc71}.class-2{border-color:#f1c40f}
 .class-3{border-color:#e67e22}.class-4{border-color:#e74c3c}.class-5{border-color:#c0392b}
 .pre{background:#f0f8ff;padding:0.8rem;border-radius:10px;border-left:5px solid #1f77b4;margin:0.5rem 0}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255,255,255)); bg.paste(img, mask=img.split()[-1]); img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def draw_square_overlay(pil_img: Image.Image, cx: int, cy: int, size: int, color=(0,255,0), thickness=3) -> np.ndarray:
    img = pil_to_rgb_np(pil_img).copy(); h,w = img.shape[:2]; half = size//2
    x1,y1 = max(0,cx-half), max(0,cy-half); x2,y2 = min(w-1,cx+half), min(h-1,cy+half)
    bgr = img[:,:,::-1].copy(); cv2.rectangle(bgr,(x1,y1),(x2,y2),(color[2],color[1],color[0]),thickness)
    return bgr[:,:,::-1]

def crop_square(pil_img: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    img = pil_to_rgb_np(pil_img); h,w = img.shape[:2]; half = size//2
    x1,y1 = max(0,cx-half), max(0,cy-half); x2,y2 = min(w,cx+half), min(h,cy+half)
    return Image.fromarray(img[y1:y2, x1:x2])

# ------------------------------------------------------------
# Ön işleme (görsel)
# ------------------------------------------------------------
def enhanced_preprocessing(image):
    if isinstance(image, Image.Image):
        image = pil_to_rgb_np(image)
    steps = {}
    # kareye pad + yeniden boyut
    h,w = image.shape[:2]
    if h!=w:
        S=max(h,w); pad_h=(S-h)//2; pad_w=(S-w)//2
        image = cv2.copyMakeBorder(image, pad_h,pad_h,pad_w,pad_w, cv2.BORDER_CONSTANT, value=[255,255,255])
    image = cv2.resize(image,(400,400)); steps["original"]=image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY); steps["grayscale"]=gray
    den = cv2.fastNlMeansDenoising(gray); steps["denoised"]=den
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cla = clahe.apply(den); steps["contrast_enhanced"]=cla
    bil = cv2.bilateralFilter(cla, 9, 75, 75); steps["bilateral"]=bil
    kernel = np.ones((3,3), np.uint8)
    opn = cv2.morphologyEx(bil, cv2.MORPH_OPEN, kernel); steps["morphology"]=opn
    adp = cv2.adaptiveThreshold(opn,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2); steps["adaptive_threshold"]=adp
    hk = cv2.getStructuringElement(cv2.MORPH_RECT,(25,1)); vk = cv2.getStructuringElement(cv2.MORPH_RECT,(1,25))
    hlines = cv2.morphologyEx(adp, cv2.MORPH_OPEN, hk); vlines = cv2.morphologyEx(adp, cv2.MORPH_OPEN, vk)
    gl = cv2.addWeighted(hlines,0.5,vlines,0.5,0); steps["grid_lines"]=gl
    fin = cv2.addWeighted(opn,0.8,gl,0.2,0); steps["final_processed"]=fin
    model_input = np.expand_dims(cv2.cvtColor(fin,cv2.COLOR_GRAY2RGB).astype(np.float32)/255.0, axis=0)
    return model_input, steps

# ------------------------------------------------------------
# Grid bölgesi (fallback'li)
# ------------------------------------------------------------
def detect_crosscut_grid_region(image):
    if isinstance(image, Image.Image): image = pil_to_rgb_np(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    edges = cv2.Canny(gray,50,150); lines = cv2.HoughLines(edges,1,np.pi/180,50)
    if lines is None:
        h,w = gray.shape; return {'x':w//4,'y':h//4,'width':w//2,'height':h//2,'detected':False}
    horiz, vert = [], []
    for ln in lines:
        rho,th = ln[0]
        if abs(th)<np.pi/4 or abs(th-np.pi)<np.pi/4: horiz.append(rho)
        elif abs(th-np.pi/2)<np.pi/4: vert.append(rho)
    if len(horiz)<4 or len(vert)<4:
        h,w = gray.shape; return {'x':w//4,'y':h//4,'width':w//2,'height':h//2,'detected':False}
    horiz.sort(); vert.sort()
    min_x,max_x = int(abs(min(vert))), int(abs(max(vert)))
    min_y,max_y = int(abs(min(horiz))), int(abs(max(horiz)))
    gw,gh = max_x-min_x, max_y-min_y
    return {'x':max(0,min_x-20),'y':max(0,min_y-20),'width':min(gray.shape[1]-max(0,min_x-20),gw+40),
            'height':min(gray.shape[0]-max(0,min_y-20),gh+40),'detected':True}

# ------------------------------------------------------------
# HAT (çizgi) hasarı analizi: 5x5 kenar boşlukları
# ------------------------------------------------------------
def analyze_line_damage_5x5(img_rgb, spacing_mm=1, lines_white=True, side_gap_thr=0.20, band_mm=0.7,
                            return_debug=False):
    if isinstance(img_rgb, Image.Image): img_rgb = pil_to_rgb_np(img_rgb)
    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim==3 else img_rgb

    # ROI
    reg = detect_crosscut_grid_region(img_rgb)
    x,y,w,h = reg['x'], reg['y'], reg['width'], reg['height']
    roi = gray_full[y:y+h, x:x+w]
    H,W = roi.shape
    cell_h, cell_w = max(1,H//5), max(1,W//5)

    # px/mm tahmini ve bant kalınlığı
    px_per_mm = max(1.0, ((cell_w+cell_h)/2.0)/float(spacing_mm))
    band_px = int(np.clip(round(band_mm*px_per_mm), 2, 14))

    # Adaptif eşik: çizgileri > present mask
    blur = cv2.GaussianBlur(roi,(3,3),0)
    binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # çizgiler beyaz mı?
    line_present = binary if lines_white else cv2.bitwise_not(binary)  # 255 = çizgi var
    line_present = cv2.morphologyEx(line_present, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))  # küçük boşlukları kapat

    # Hücre/kenar bazlı ölçüm
    contrib_map = np.zeros((5,5), dtype=np.float32)
    debug_imgs = {}

    for i in range(5):
        for j in range(5):
            y1, y2 = i*cell_h, min(H,(i+1)*cell_h)
            x1, x2 = j*cell_w, min(W,(j+1)*cell_w)

            # Kenar bantları
            bands = {
                "top":  (slice(max(0,y1-band_px//2), min(H,y1+band_px//2)), slice(x1,x2)),
                "bot":  (slice(max(0,y2-band_px//2), min(H,y2+band_px//2)), slice(x1,x2)),
                "left": (slice(y1,y2), slice(max(0,x1-band_px//2), min(W,x1+band_px//2))),
                "right":(slice(y1,y2), slice(max(0,x2-band_px//2), min(W,x2+band_px//2)))
            }
            side_scores=[]
            for slc in bands.values():
                band = np.ones((roi[slc].shape[0], roi[slc].shape[1]), dtype=np.uint8)*255  # beklenen çizgi alanı
                present = line_present[slc]  # 255 = çizgi var
                # boşluk (hasar) = beklenen bandta çizgi YOK (0)
                # present 255 ise çizgi var; 0 ise boşluk
                gap = (present==0).astype(np.uint8)*255
                area = band.size
                gap_ratio = float(np.sum(gap>0))/float(area) if area>0 else 0.0
                side_scores.append(min(1.0, gap_ratio/side_gap_thr))
            contrib_map[i,j] = float(np.mean(side_scores))  # 0..1

    damaged_cells_eq = float(np.sum(contrib_map))  # 0..25
    damaged_cells_bin = float(np.sum(contrib_map >= 0.25))  # en az %25 katkısı olanları 1 say
    grid_detected = True  # ROI çıkarıldı varsayıyoruz

    out = {
        "contrib_map": contrib_map.tolist(),
        "damaged_cells_eq": damaged_cells_eq,
        "damaged_cells_binary": damaged_cells_bin,
        "total_cells": 25,
        "px_per_mm": px_per_mm,
        "band_px": band_px,
        "cell_h": cell_h, "cell_w": cell_w,
        "grid_region": {'x':int(x),'y':int(y),'width':int(w),'height':int(h)},
        "grid_detected": grid_detected,
        "side_gap_thr": float(side_gap_thr),
        "lines_white": bool(lines_white)
    }

    if return_debug:
        out["debug"] = {
            "roi": roi,
            "binary": binary,
            "line_present": line_present
        }
    return out

# ------------------------------------------------------------
# Sınıf eşleme + olasılık
# ------------------------------------------------------------
def map_cells_to_class(n_eq: float) -> int:
    if n_eq <= 0.0: return 0
    elif n_eq <= 1.25: return 1
    elif n_eq <= 3.5: return 2
    elif n_eq <= 8.5: return 3
    elif n_eq <= 16.5: return 4
    else: return 5

def rule_based_probabilities(n_eq: float) -> np.ndarray:
    dom = map_cells_to_class(n_eq)
    sev = np.clip(n_eq/25.0, 0.0, 1.0)
    main = 0.60 + 0.35*sev; side = (1.0-main)/2.0
    p = np.zeros(6, np.float32); p[dom]=main
    if dom-1>=0: p[dom-1]=side
    if dom+1<=5: p[dom+1]=side
    return p/p.sum()

# ------------------------------------------------------------
# UI ve akış
# ------------------------------------------------------------
def main():
    st.markdown('<div class="main-header"><h1>🔬 ISO 2409 – Hat Hasarı ile Sınıflandırma</h1>'
                '<p>5×5 hücre – çizgi (hat) üzerindeki kopmalara göre karar</p></div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Ayarlar")
    spacing_mm = st.sidebar.radio("Kesik aralığı (mm)", [1,2,3], index=0, horizontal=True)
    mode = st.sidebar.radio("Karar modu", ["Hat hasarı (önerilen)","Adaptif piksel hasarı"], index=0)
    lines_white = st.sidebar.checkbox("Kesik çizgileri beyaz mı?", value=True,
                                      help="Örn. kırmızı zeminde beyaz çizgiler → işaretli kalsın")
    side_gap_thr = st.sidebar.slider("Kenar boşluk eşiği", 0.05, 0.6, 0.20, 0.05,
                                     help="Bir kenarın 'hasarlı' sayılması için boşluk oranı")
    band_mm = st.sidebar.slider("Kenar bant kalınlığı (mm)", 0.3, 1.5, 0.7, 0.1,
                                help="Çizgi etrafında incelenecek şerit kalınlığı")
    show_debug = st.sidebar.checkbox("Debug görsellerini göster", value=False)

    st.sidebar.markdown("---")
    st.sidebar.info("Sınıf eşleme: 0→C0, (0–1.25]→C1, (1.25–3.5]→C2, (3.5–8.5]→C3, (8.5–16.5]→C4, >16.5→C5")

    col1,col2 = st.columns([1.2,1])

    with col1:
        st.subheader("Görüntü Yükleme")
        up = st.file_uploader("Çapraz kesim görüntüsü yükle (png/jpg/jpeg)", type=["png","jpg","jpeg"])
        if up is not None:
            img = Image.open(up); st.image(img, caption="Yüklenen Görüntü")

            # Kare seçim
            w,h = img.size
            auto = detect_crosscut_grid_region(img)
            auto_size = int(min(auto['width'], auto['height'])) if auto else int(min(w,h)*0.6)
            auto_cx = int(auto['x'] + auto['width']//2) if auto else w//2
            auto_cy = int(auto['y'] + auto['height']//2) if auto else h//2

            c1,c2,c3 = st.columns(3)
            with c1: cx = st.slider("Merkez X", 0, w-1, auto_cx, 1)
            with c2: cy = st.slider("Merkez Y", 0, h-1, auto_cy, 1)
            with c3: size = st.slider("Kare Boyutu", 50, min(w,h), auto_size, 5)

            st.image(draw_square_overlay(img, cx, cy, size), caption="Ön izleme")

            if st.button("Kırp ve Analiz Et", type="primary"):
                crop = crop_square(img, cx, cy, size)
                st.image(crop, caption=f"Kırpılmış ({size}x{size})")

                st.subheader("Kontrast/Parlaklık")
                ct = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1)
                br = st.slider("Parlaklık", 0.5, 2.0, 1.0, 0.1)
                final = crop
                if ct!=1.0 or br!=1.0:
                    e = ImageEnhance.Contrast(crop); tmp = e.enhance(ct)
                    e = ImageEnhance.Brightness(tmp); tmp = e.enhance(br)
                    st.image(tmp, caption=f"Ayarlanmış (K:{ct:.1f}, P:{br:.1f})")
                    final = tmp

                if mode == "Hat hasarı (önerilen)":
                    g = analyze_line_damage_5x5(final, spacing_mm=int(spacing_mm),
                                                lines_white=bool(lines_white),
                                                side_gap_thr=float(side_gap_thr),
                                                band_mm=float(band_mm),
                                                return_debug=bool(show_debug))
                    n_eq = g["damaged_cells_eq"]
                else:
                    # Basit yedek: adaptif piksel hasarı (hücre içi; çizgileri dikkate almaz)
                    _, steps = enhanced_preprocessing(final)
                    adp = steps["adaptive_threshold"]
                    H,W = adp.shape; cell_h,cell_w = H//5, W//5
                    # siyah pikselleri "hasar" say (gerekirse invert et)
                    dmg = (adp==0).astype(np.uint8)
                    contrib = []
                    thr = 0.003
                    for i in range(5):
                        for j in range(5):
                            y1,y2 = i*cell_h, (i+1)*cell_h; x1,x2 = j*cell_w, (j+1)*cell_w
                            cell = dmg[y1:y2, x1:x2]
                            r = cell.mean()  # 0..1
                            contrib.append(min(1.0, r/max(thr,1e-6)))
                    n_eq = float(np.sum(contrib))
                    g = {"contrib_map": np.array(contrib).reshape(5,5).tolist(),
                         "grid_region":{"x":0,"y":0,"width":W,"height":H},
                         "band_px":0,"px_per_mm":0,"grid_detected":True}

                cls = map_cells_to_class(n_eq)
                probs = rule_based_probabilities(n_eq)
                st.session_state.result = {"n_eq":n_eq, "cls":cls, "probs":probs, "g":g, "mode":mode, "lines_white":lines_white, "side_gap_thr":side_gap_thr}

    with col2:
        st.subheader("📊 Sonuçlar")
        if "result" in st.session_state:
            r = st.session_state.result; cls = r["cls"]; n_eq = r["n_eq"]; probs = r["probs"]; g = r["g"]
            iso_classes = {
                0: {"name":"Sınıf 0","quality":"Mükemmel","desc":"Kopma yok","color":"#27ae60"},
                1: {"name":"Sınıf 1","quality":"Çok İyi","desc":"Çok küçük/seyrek kopmalar","color":"#2ecc71"},
                2: {"name":"Sınıf 2","quality":"İyi","desc":"Kenar boyunca küçük kopmalar","color":"#f1c40f"},
                3: {"name":"Sınıf 3","quality":"Kabul","desc":"Belirgin kopmalar","color":"#e67e22"},
                4: {"name":"Sınıf 4","quality":"Zayıf","desc":"Alan >~%5 etkilenmiş","color":"#e74c3c"},
                5: {"name":"Sınıf 5","quality":"Çok Zayıf","desc":"Yaygın kopma","color":"#c0392b"},
            }
            info = iso_classes[cls]
            st.markdown(f'<div class="prediction-box class-{cls}" style="border-color:{info["color"]}">'
                        f'<h2>{info["name"]}</h2><h3>{info["quality"]}</h3><p>{info["desc"]}</p>'
                        f'<p><em>Karar modu: {r["mode"]}</em></p></div>', unsafe_allow_html=True)
            st.metric("Eşdeğer hasarlı hücre", f"{n_eq:.2f}/25")
            df = pd.DataFrame({"Sınıf":[f"Sınıf {i}" for i in range(6)], "Olasılık":[float(x)*100 for x in probs],
                               "Renk":[iso_classes[i]["color"] for i in range(6)]})
            fig = px.bar(df, x="Sınıf", y="Olasılık", color="Renk",
                         color_discrete_map={c:c for c in df["Renk"]},
                         title="Sınıf Olasılıkları")
            fig.update_layout(showlegend=False, height=360, yaxis_title="Olasılık (%)")
            st.plotly_chart(fig, use_container_width=True)

            # Hücre 5x5 ısı haritası
            contrib_map = np.array(g["contrib_map"], dtype=float)
            hm = px.imshow(contrib_map, origin="upper", text_auto=True, range_color=[0,1], aspect="equal",
                           title="Hücre katkı haritası (0..1)")
            st.plotly_chart(hm, use_container_width=True)

            if show_debug and "debug" in g:
                st.markdown("### 🧪 Debug")
                st.image(g["debug"]["roi"], caption="ROI (Gri)")
                st.image(g["debug"]["binary"], caption="Adaptif Binary")
                st.image(g["debug"]["line_present"], caption="Çizgi (var) maskesi")

    st.caption("Not: 'Hat hasarı' modu, hücre kenar bantlarında çizgi boşluklarını ölçer; hücre içindeki noktasal gürültüyü saymaz.")

# ------------------------------------------------------------
# Çalıştırma
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
