# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageEnhance

# ========= UI & Stil =========
st.set_page_config(page_title="ISO 2409 – Hücre İçi Kopma (Datasetsiz, renk-farkındalıklı)", page_icon="🔬", layout="wide")
st.markdown("""
<style>
 .hdr{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center;margin-bottom:1rem}
 .card{background:#f8f9fa;padding:1.2rem;border-radius:15px;border:3px solid;text-align:center;margin:1rem 0}
 .class-0{border-color:#27ae60}.class-1{border-color:#2ecc71}.class-2{border-color:#f1c40f}
 .class-3{border-color:#e67e22}.class-4{border-color:#e74c3c}.class-5{border-color:#c0392b}
</style>
""", unsafe_allow_html=True)

# ========= Yardımcılar =========
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255,255,255)); bg.paste(img, mask=img.split()[-1]); img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def draw_square_overlay(pil_img: Image.Image, cx: int, cy: int, size: int) -> np.ndarray:
    img = pil_to_rgb_np(pil_img).copy(); h,w = img.shape[:2]; half = size//2
    x1,y1 = max(0,cx-half), max(0,cy-half); x2,y2 = min(w-1,cx+half), min(h-1,cy+half)
    bgr = img[:,:,::-1].copy(); cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,255,0),3)
    return bgr[:,:,::-1]

def crop_square(pil_img: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    img = pil_to_rgb_np(pil_img); h,w = img.shape[:2]; half=size//2
    x1,y1 = max(0,cx-half), max(0,cy-half); x2,y2 = min(w,cx+half), min(h,cy+half)
    return Image.fromarray(img[y1:y2, x1:x2])

def detect_crosscut_grid_region(image):
    if isinstance(image, Image.Image): image = pil_to_rgb_np(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 70)
    if lines is None:
        h,w = gray.shape; return {'x':w//4,'y':h//4,'width':w//2,'height':h//2}
    horiz, vert = [], []
    for ln in lines:
        rho,th = ln[0]
        if abs(th)<np.pi/4 or abs(th-np.pi)<np.pi/4: horiz.append(rho)
        elif abs(th-np.pi/2)<np.pi/4: vert.append(rho)
    if len(horiz)<4 or len(vert)<4:
        h,w = gray.shape; return {'x':w//4,'y':h//4,'width':w//2,'height':h//2}
    horiz.sort(); vert.sort()
    min_x,max_x = int(abs(min(vert))), int(abs(max(vert)))
    min_y,max_y = int(abs(min(horiz))), int(abs(max(horiz)))
    gw,gh = max_x-min_x, max_y-min_y
    return {'x':max(0,min_x-20),'y':max(0,min_y-20),'width':gw+40,'height':gh+40}

# --- Renk-farkındalıklı çizgi ve kopma maskeleri ---
def color_aware_masks(grid_rgb: np.ndarray, s_white_thr=60, v_white_thr=170, v_dark_thr=70):
    """
    Dönüş:
      line_mask (255=çizgi), damage_candidate (255=hücre içi kopma adayı)
    Mantık:
      - Renkli zeminde çizgiler genelde beyaz/soluk: S düşük, V yüksek  -> line_mask
      - Kopma: renkli zeminde **beyaz lekeler** (white_mask - lines)
               beyaz zeminde **koyu lekeler** (dark_mask)
    """
    hsv = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv)

    # Beyaz/soluk maskesi (çizgiler + olası kopma lekeleri)
    white_mask = ((S <= s_white_thr) & (V >= v_white_thr)).astype(np.uint8)*255

    # Beyaz zeminde koyu lekeleri yakalamak için
    dark_mask = (V <= v_dark_thr).astype(np.uint8)*255

    # Satürasyon ortalamasına göre zemin tipi
    mean_S, mean_V = float(np.mean(S)), float(np.mean(V))
    is_white_surface = (mean_S < 40 and mean_V > 180)

    # Çizgi maskesini ince-uzun yapıları vurgulayarak rafine et
    h, w = white_mask.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w//16), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, h//16)))
    line_mask = cv2.max(cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, hk),
                        cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, vk))

    # Kopma adayı
    if is_white_surface:
        damage_candidate = dark_mask.copy()
    else:
        damage_candidate = white_mask.copy()  # beyaz lekeler kopma adayı

    # Çizgileri tamamen çıkar
    damage_candidate = cv2.bitwise_and(damage_candidate, cv2.bitwise_not(line_mask))

    # Gürültü küçült
    damage_candidate = cv2.morphologyEx(damage_candidate, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    return line_mask, damage_candidate, {"is_white_surface": is_white_surface, "mean_S":mean_S, "mean_V":mean_V}

def make_cut_mask(gray, px_per_mm, extra_dilate=1):
    # Hough + morfoloji birleşik
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=max(20, gray.shape[1]//3), maxLineGap=10)
    mask = np.zeros_like(gray, np.uint8)
    thick = int(np.clip(round(0.35*px_per_mm), 2, 18))
    if lines is not None:
        for l in lines[:,0]:
            cv2.line(mask, (l[0],l[1]), (l[2],l[3]), 255, thick)
    # renk tabanlı çizgileri de ekle (gri üzerinden tahmin zor olabilir; RGB ile tekrar bakılacak)
    if extra_dilate>0:
        mask = cv2.dilate(mask, np.ones((3,3),np.uint8), extra_dilate)
    return mask

# ========= ÇEKİRDEK: Hücre İÇİ kopma analizi =========
def analyze_cell_damage_5x5(
    img_rgb,
    spacing_mm=1,
    s_white_thr=60,
    v_white_thr=170,
    v_dark_thr=70,
    min_pix_ratio=0.0015,
    cell_ratio_thr=0.004,
    use_adaptive_fallback=True,
    return_debug=False
):
    if isinstance(img_rgb, Image.Image): img_rgb = pil_to_rgb_np(img_rgb)
    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ROI
    reg = detect_crosscut_grid_region(img_rgb)
    x,y,w,h = reg['x'], reg['y'], reg['width'], reg['height']
    grid_gray = gray_full[y:y+h, x:x+w]
    grid_rgb  = img_rgb[y:y+h, x:x+w]
    H,W = grid_gray.shape
    cell_h, cell_w = max(1,H//5), max(1,W//5)
    px_per_mm = max(1.0, ((cell_w+cell_h)/2.0)/float(spacing_mm))

    # 1) Renk-farkındalıklı çizgi & kopma aday maskeleri
    line_mask_rgb, damage_candidate_rgb, surf_info = color_aware_masks(grid_rgb, s_white_thr, v_white_thr, v_dark_thr)

    # 2) Hough tabanlı çizgi maskesi ile birleşim
    line_mask_hough = make_cut_mask(grid_gray, px_per_mm, extra_dilate=1)
    line_mask = cv2.max(line_mask_rgb, line_mask_hough)

    # 3) Yalnızca hücre içi (çizgiler hariç) kopma
    interior_mask = cv2.bitwise_not(line_mask)
    flake_interior = cv2.bitwise_and(damage_candidate_rgb, interior_mask)

    # 4) (Opsiyonel) Fallback: çok düşük sinyal varsa adaptif eşik dene
    if use_adaptive_fallback and np.mean(flake_interior>0) < 0.001:
        adp = cv2.adaptiveThreshold(grid_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # beyaz zeminse koyu, değilse beyaz lekeler
        if surf_info["is_white_surface"]:
            cand2 = (adp==0).astype(np.uint8)*255
        else:
            cand2 = adp.copy()  # adp (THRESH_BINARY) beyazı öne çıkarır
        flake_interior = cv2.bitwise_and(cand2, interior_mask)
        flake_interior = cv2.morphologyEx(flake_interior, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # 5) Hücre bazında fraksiyonel katkı
    cell_area = cell_h*cell_w
    MIN_PIX = max(6, int(min_pix_ratio*cell_area))
    THR = float(cell_ratio_thr)
    contrib_map = np.zeros((5,5), np.float32)
    for i in range(5):
        for j in range(5):
            y1,y2 = i*cell_h, min(H,(i+1)*cell_h)
            x1,x2 = j*cell_w, min(W,(j+1)*cell_w)
            cell_int = interior_mask[y1:y2, x1:x2]
            cell_flk = flake_interior[y1:y2, x1:x2]
            area = max(1, int(np.sum(cell_int>0)))
            flk  = int(np.sum(cell_flk>0))
            ratio = flk/area
            contrib = 0.0
            if flk >= MIN_PIX:
                contrib = min(1.0, ratio/max(THR,1e-6))
            contrib_map[i,j] = contrib

    damaged_cells_eq  = float(np.sum(contrib_map))              # 0..25
    damaged_cells_bin = float(np.sum(contrib_map>=1.0))
    delam_ratio_pct  = float(np.mean(flake_interior>0))*100.0

    out = {
        "contrib_map": contrib_map.tolist(),
        "damaged_cells_eq": damaged_cells_eq,
        "damaged_cells_binary": damaged_cells_bin,
        "delamination_ratio_pct": delam_ratio_pct,
        "grid_region": {"x":int(x),"y":int(y),"width":int(w),"height":int(h)},
        "px_per_mm": px_per_mm,
        "surface_info": surf_info
    }
    if return_debug:
        out["debug"] = {
            "grid_rgb": grid_rgb, "grid_gray": grid_gray,
            "line_mask_rgb": line_mask_rgb, "line_mask_hough": line_mask_hough, "line_mask": line_mask,
            "damage_candidate_rgb": damage_candidate_rgb, "flake_interior": flake_interior
        }
    return out

def map_cells_to_class(n_eq: float) -> int:
    if n_eq <= 0.0:   return 0
    if n_eq <= 1.25:  return 1
    if n_eq <= 3.5:   return 2
    if n_eq <= 8.5:   return 3
    if n_eq <= 16.5:  return 4
    return 5

# ========= UI =========
st.markdown('<div class="hdr"><h1>🔬 ISO 2409 – Hücre İçi Kopma (Renk-farkındalıklı)</h1>'
            '<p>Çizgi (hat) pikselleri güvenle maskelenir; yalnızca hücre içi beyaz/koyu lekeler sayılır</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Ayarlar")
    spacing_mm     = st.radio("Kesik aralığı (mm)", [1,2,3], index=0, horizontal=True)
    s_white_thr    = st.slider("S (HSV) beyaz eşiği", 10, 120, 60, 5)
    v_white_thr    = st.slider("V (HSV) beyaz eşiği", 80, 240, 170, 5)
    v_dark_thr     = st.slider("V (HSV) koyu eşiği",  10, 120, 70, 5)
    cell_ratio_thr = st.slider("Hücre hasar oran eşiği", 0.001, 0.02, 0.004, 0.001)
    min_pix_ratio  = st.slider("Min piksel (hücre oranı)", 0.0005, 0.02, 0.0015, 0.0005)
    use_adp_fb     = st.checkbox("Sinyal çok azsa adaptif eşik fallback", value=True)
    show_debug     = st.checkbox("Debug görsellerini göster", value=False)

col1, col2 = st.columns([1.2,1])

with col1:
    st.subheader("Görüntü Yükle")
    up = st.file_uploader("png/jpg/jpeg", type=["png","jpg","jpeg"])
    if up:
        img = Image.open(up); st.image(img, caption="Yüklenen")
        # ROI + manuel ayar
        W,H = img.size
        auto = detect_crosscut_grid_region(img)
        auto_size = int(min(auto['width'], auto['height'])) if auto else int(min(W,H)*0.6)
        auto_cx   = int(auto['x'] + auto['width']//2) if auto else W//2
        auto_cy   = int(auto['y'] + auto['height']//2) if auto else H//2
        c1,c2,c3 = st.columns(3)
        with c1: cx = st.slider("Merkez X", 0, W-1, auto_cx, 1)
        with c2: cy = st.slider("Merkez Y", 0, H-1, auto_cy, 1)
        with c3: size = st.slider("Kare Boyutu", 50, min(W,H), auto_size, 5)
        st.image(draw_square_overlay(img, cx, cy, size), caption="Ön izleme")

        if st.button("Kırp ve Analiz Et", type="primary"):
            crop = crop_square(img, cx, cy, size); st.image(crop, caption=f"Kırpılmış ({size}x{size})")

            st.subheader("Kontrast/Parlaklık")
            ct = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1); br = st.slider("Parlaklık", 0.5, 2.0, 1.0, 0.1)
            final = crop
            if ct!=1.0 or br!=1.0:
                e = ImageEnhance.Contrast(crop); tmp = e.enhance(ct)
                e = ImageEnhance.Brightness(tmp);  tmp = e.enhance(br)
                st.image(tmp, caption=f"Ayarlanmış (K:{ct:.1f}, P:{br:.1f})")
                final = tmp

            g = analyze_cell_damage_5x5(
                final,
                spacing_mm=int(spacing_mm),
                s_white_thr=int(s_white_thr),
                v_white_thr=int(v_white_thr),
                v_dark_thr=int(v_dark_thr),
                min_pix_ratio=float(min_pix_ratio),
                cell_ratio_thr=float(cell_ratio_thr),
                use_adaptive_fallback=bool(use_adp_fb),
                return_debug=bool(show_debug)
            )
            n_eq = g["damaged_cells_eq"]; cls = map_cells_to_class(n_eq)
            st.session_state.result = {"g":g, "n_eq":n_eq, "cls":cls}

with col2:
    st.subheader("📊 Sonuçlar")
    if "result" in st.session_state:
        r = st.session_state.result; g=r["g"]; n_eq=r["n_eq"]; cls=r["cls"]
        classes = {
            0: {"title":"Sınıf 0","q":"Mükemmel","desc":"Hücre içinde kopma yok","color":"#27ae60"},
            1: {"title":"Sınıf 1","q":"Çok İyi","desc":"Çok küçük/seyrek kopmalar","color":"#2ecc71"},
            2: {"title":"Sınıf 2","q":"İyi","desc":"Kenar boyunca küçük kopmalar","color":"#f1c40f"},
            3: {"title":"Sınıf 3","q":"Kabul","desc":"Belirgin kopmalar","color":"#e67e22"},
            4: {"title":"Sınıf 4","q":"Zayıf","desc":"Alan >~%5 etkilenmiş","color":"#e74c3c"},
            5: {"title":"Sınıf 5","q":"Çok Zayıf","desc":"Yaygın kopma","color":"#c0392b"},
        }
        info = classes[cls]
        st.markdown(f'<div class="card class-{cls}" style="border-color:{info["color"]}">'
                    f'<h2>{info["title"]}</h2><h3>{info["q"]}</h3><p>{info["desc"]}</p></div>', unsafe_allow_html=True)
        cA,cB,cC = st.columns(3)
        with cA: st.metric("Eşdeğer hasarlı hücre", f"{n_eq:.2f}/25")
        with cB: st.metric("Binary hasarlı hücre",  f"{g["damaged_cells_binary"]:.0f}/25")
        with cC: st.metric("Delaminasyon (pikselle)", f"{g["delamination_ratio_pct"]:.2f}%")

        contrib = np.array(g["contrib_map"], float)
        fig = px.imshow(contrib, origin="upper", text_auto=True, range_color=[0,1], aspect="equal",
                        title="Hücre içi hasar katkısı (0..1)")
        st.plotly_chart(fig, use_container_width=True)

        if show_debug and "debug" in g:
            st.markdown("### 🧪 Debug")
            dbg = g["debug"]
            st.image(dbg["grid_rgb"], caption="ROI (RGB)")
            st.image(dbg["grid_gray"], caption="ROI (Gri)")
            st.image(dbg["line_mask_rgb"], caption="Çizgi maskesi (renk tabanlı)")
            st.image(dbg["line_mask_hough"], caption="Çizgi maskesi (Hough)")
            st.image(dbg["line_mask"], caption="Birleşik çizgi maskesi")
            st.image(dbg["damage_candidate_rgb"], caption="Kopma adayı (renk tabanlı)")
            st.image(dbg["flake_interior"], caption="Kesik HARIÇ kopma")

st.caption("Kural aralıkları: 0→C0, (0–1.25]→C1, (1.25–3.5]→C2, (3.5–8.5]→C3, (8.5–16.5]→C4, >16.5→C5. \
Bu sürüm, kırmızı zemin gibi durumlarda çizgileri **renkten** de ayırdığı için temiz örneklerde Class 0 döner.")
