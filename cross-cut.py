# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageEnhance

# ---------- UI ----------
st.set_page_config(page_title="ISO 2409 – Hücre İçi Kopma ile Sınıflandırma", page_icon="🔬", layout="wide")

st.markdown("""
<style>
 .hdr{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center;margin-bottom:1rem}
 .box{background:#f8f9fa;padding:1.2rem;border-radius:15px;border:3px solid;text-align:center;margin:1rem 0}
 .class-0{border-color:#27ae60}.class-1{border-color:#2ecc71}.class-2{border-color:#f1c40f}
 .class-3{border-color:#e67e22}.class-4{border-color:#e74c3c}.class-5{border-color:#c0392b}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255,255,255)); bg.paste(img, mask=img.split()[-1]); img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def draw_square_overlay(pil_img: Image.Image, cx: int, cy: int, size: int) -> np.ndarray:
    img = pil_to_rgb_np(pil_img).copy(); h,w = img.shape[:2]; half=size//2
    x1,y1 = max(0,cx-half), max(0,cy-half); x2,y2 = min(w-1,cx+half), min(h-1,cy+half)
    bgr = img[:,:,::-1].copy(); cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,255,0),3)
    return bgr[:,:,::-1]

def crop_square(pil_img: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    img = pil_to_rgb_np(pil_img); h,w = img.shape[:2]; half=size//2
    x1,y1 = max(0,cx-half), max(0,cy-half); x2,y2 = min(w,cx+half), min(h,cy+half)
    return Image.fromarray(img[y1:y2, x1:x2])

def detect_crosscut_grid_region(image):
    if isinstance(image, Image.Image): image = pil_to_rgb_np(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
    if lines is None:
        h,w = gray.shape; return {'x':w//4,'y':h//4,'width':w//2,'height':h//2}
    horiz, vert = [], []
    for ln in lines:
        rho,theta = ln[0]
        if abs(theta)<np.pi/4 or abs(theta-np.pi)<np.pi/4: horiz.append(rho)
        elif abs(theta-np.pi/2)<np.pi/4: vert.append(rho)
    if len(horiz)<4 or len(vert)<4:
        h,w = gray.shape; return {'x':w//4,'y':h//4,'width':w//2,'height':h//2}
    horiz.sort(); vert.sort()
    min_x,max_x = int(abs(min(vert))), int(abs(max(vert)))
    min_y,max_y = int(abs(min(horiz))), int(abs(max(horiz)))
    gw,gh = max_x-min_x, max_y-min_y
    return {'x':max(0,min_x-20),'y':max(0,min_y-20),'width':gw+40,'height':gh+40}

def make_cut_mask(grid_gray: np.ndarray, thickness_px: int) -> np.ndarray:
    edges = cv2.Canny(grid_gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                            minLineLength=max(20, grid_gray.shape[1]//3), maxLineGap=10)
    mask = np.zeros_like(grid_gray, dtype=np.uint8)
    if lines is not None:
        for l in lines[:,0]:
            cv2.line(mask, (l[0],l[1]), (l[2],l[3]), 255, max(1,int(thickness_px)))
    # biraz genişlet ki çizgi kenarı da dışarıda kalsın
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8), 1)
    return mask

# ---------- Core: Hücre içi hasar (adaptif eşik + çizgi maskesi) ----------
def analyze_cell_damage_5x5(
    img_rgb,
    spacing_mm=1,
    invert_adaptive=False,
    cell_ratio_thr=0.003,
    min_pix_ratio=0.001,
    return_debug=False
):
    if isinstance(img_rgb, Image.Image): img_rgb = pil_to_rgb_np(img_rgb)
    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim==3 else img_rgb

    # ROI (grid bölgesi)
    reg = detect_crosscut_grid_region(img_rgb)
    x,y,w,h = reg['x'], reg['y'], reg['width'], reg['height']
    grid_gray = gray_full[y:y+h, x:x+w]
    H,W = grid_gray.shape
    cell_h, cell_w = max(1,H//5), max(1,W//5)

    # px/mm ve kesik kalınlığı
    px_per_mm = max(1.0, ((cell_w+cell_h)/2.0)/float(spacing_mm))
    cut_thick = int(np.clip(round(0.35*px_per_mm), 2, 14))

    # Kesik maskesi (çizgileri hariç tutacağız)
    cut_mask = make_cut_mask(grid_gray, cut_thick)
    interior_mask = cv2.bitwise_not(cut_mask)

    # Adaptif eşik (zarar = siyah pikseller)
    mode = cv2.THRESH_BINARY_INV if not invert_adaptive else cv2.THRESH_BINARY
    adaptive = cv2.adaptiveThreshold(grid_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,mode,11,2)
    # zarar maskesi (255 = zarar)
    flake_mask = adaptive
    # yalnızca hücre içini say (çizgiler hariç)
    flake_interior = cv2.bitwise_and(flake_mask, interior_mask)

    # Eşikler
    cell_area = cell_h*cell_w
    MIN_PIX = max(5, int(min_pix_ratio*cell_area))
    THR = float(cell_ratio_thr)

    # Hücre bazında fraksiyonel katkı
    contrib_map = np.zeros((5,5), dtype=np.float32)
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

    damaged_cells_eq = float(np.sum(contrib_map))  # 0..25
    damaged_cells_bin = float(np.sum(contrib_map>=1.0))

    out = {
        "contrib_map": contrib_map.tolist(),
        "damaged_cells_eq": damaged_cells_eq,
        "damaged_cells_binary": damaged_cells_bin,
        "grid_region": {"x":int(x),"y":int(y),"width":int(w),"height":int(h)},
        "px_per_mm": px_per_mm,
        "cut_thickness_px": cut_thick,
        "cell_area_px": cell_area,
        "cell_ratio_thr": THR,
        "min_pix": MIN_PIX
    }
    if return_debug:
        out["debug"] = {
            "grid_gray": grid_gray,
            "cut_mask": cut_mask,
            "interior_mask": interior_mask,
            "adaptive_binary": adaptive,
            "flake_interior": flake_interior
        }
    return out

def map_cells_to_class(n_eq: float) -> int:
    if n_eq <= 0.0: return 0
    elif n_eq <= 1.25: return 1
    elif n_eq <= 3.5: return 2
    elif n_eq <= 8.5: return 3
    elif n_eq <= 16.5: return 4
    else: return 5

def probs_from_class(n_eq: float) -> np.ndarray:
    dom = map_cells_to_class(n_eq)
    sev = np.clip(n_eq/25.0, 0.0, 1.0)
    main = 0.60 + 0.35*sev; side = (1.0-main)/2.0
    p = np.zeros(6, np.float32); p[dom]=main
    if dom-1>=0: p[dom-1]=side
    if dom+1<=5: p[dom+1]=side
    return p/p.sum()

# ---------- UI Flow ----------
st.markdown('<div class="hdr"><h1>🔬 ISO 2409 – Hücre İçi Kopma ile Sınıflandırma</h1>'
            '<p>Kesik çizgileri hariç tutularak 5×5 hücre içindeki kopmalar sayılır</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Ayarlar")
    spacing_mm = st.radio("Kesik aralığı (mm)", [1,2,3], index=0, horizontal=True)
    cell_ratio_thr = st.slider("Hücre hasar oran eşiği", 0.001, 0.02, 0.003, 0.001,
                               help="Bir hücrenin (tam=1) sayılması için gereken oran")
    min_pix_ratio = st.slider("Min piksel (hücre alan oranı)", 0.0005, 0.01, 0.001, 0.0005,
                              help="Gürültüyü elemek için asgari hasar pikseli")
    invert_adaptive = st.checkbox("Adaptif eşiği tersle (beyaz = zarar)", value=False,
                                  help="Verinde zarar beyaz görünüyorsa aç")
    show_debug = st.checkbox("Debug görsellerini göster", value=False)

col1, col2 = st.columns([1.2,1])

with col1:
    st.subheader("Görüntü Yükleme")
    up = st.file_uploader("Çapraz kesim görüntüsü (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    if up is not None:
        img = Image.open(up); st.image(img, caption="Yüklenen Görüntü")

        # Kare seçim
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
            crop = crop_square(img, cx, cy, size)
            st.image(crop, caption=f"Kırpılmış ({size}x{size})")

            st.subheader("Kontrast/Parlaklık")
            ct = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1)
            br = st.slider("Parlaklık", 0.5, 2.0, 1.0, 0.1)
            fin = crop
            if ct!=1.0 or br!=1.0:
                e = ImageEnhance.Contrast(crop); tmp = e.enhance(ct)
                e = ImageEnhance.Brightness(tmp);  tmp = e.enhance(br)
                st.image(tmp, caption=f"Ayarlanmış (K:{ct:.1f}, P:{br:.1f})")
                fin = tmp

            g = analyze_cell_damage_5x5(
                fin,
                spacing_mm=int(spacing_mm),
                invert_adaptive=bool(invert_adaptive),
                cell_ratio_thr=float(cell_ratio_thr),
                min_pix_ratio=float(min_pix_ratio),
                return_debug=bool(show_debug)
            )
            n_eq = g["damaged_cells_eq"]
            cls = map_cells_to_class(n_eq)
            probs = probs_from_class(n_eq)
            st.session_state.result = {"g":g, "n_eq":n_eq, "cls":cls, "probs":probs}

with col2:
    st.subheader("📊 Sonuçlar")
    if "result" in st.session_state:
        r = st.session_state.result
        n_eq, cls, probs, g = r["n_eq"], r["cls"], r["probs"], r["g"]
        classes = {
            0: {"title":"Sınıf 0", "q":"Mükemmel", "desc":"Hücre içinde kopma yok", "color":"#27ae60"},
            1: {"title":"Sınıf 1", "q":"Çok İyi", "desc":"Çok küçük/seyrek kopmalar", "color":"#2ecc71"},
            2: {"title":"Sınıf 2", "q":"İyi", "desc":"Kenar boyunca küçük kopmalar", "color":"#f1c40f"},
            3: {"title":"Sınıf 3", "q":"Kabul", "desc":"Belirgin kopmalar", "color":"#e67e22"},
            4: {"title":"Sınıf 4", "q":"Zayıf", "desc":"Alan >~%5 etkilenmiş", "color":"#e74c3c"},
            5: {"title":"Sınıf 5", "q":"Çok Zayıf", "desc":"Yaygın kopma", "color":"#c0392b"},
        }
        info = classes[cls]
        st.markdown(f'<div class="box class-{cls}" style="border-color:{info["color"]}">'
                    f'<h2>{info["title"]}</h2><h3>{info["q"]}</h3><p>{info["desc"]}</p></div>', unsafe_allow_html=True)
        st.metric("Eşdeğer hasarlı hücre", f"{n_eq:.2f}/25")

        df = pd.DataFrame({"Sınıf":[f"Sınıf {i}" for i in range(6)],
                           "Olasılık":[float(x)*100 for x in probs],
                           "Renk":[classes[i]["color"] for i in range(6)]})
        fig = px.bar(df, x="Sınıf", y="Olasılık", color="Renk",
                     color_discrete_map={c:c for c in df["Renk"]}, title="Sınıf Olasılıkları")
        fig.update_layout(showlegend=False, height=360, yaxis_title="Olasılık (%)")
        st.plotly_chart(fig, use_container_width=True)

        # Hücre 5×5 katkı haritası
        contrib = np.array(g["contrib_map"], dtype=float)
        heat = px.imshow(contrib, origin="upper", text_auto=True, range_color=[0,1], aspect="equal",
                         title="Hücre içi hasar katkısı (0..1)")
        st.plotly_chart(heat, use_container_width=True)

        if show_debug and "debug" in g:
            st.markdown("### 🧪 Debug")
            st.image(g["debug"]["grid_gray"], caption="ROI (Gri)")
            st.image(g["debug"]["cut_mask"], caption="Kesik Maskesi")
            st.image(g["debug"]["interior_mask"], caption="İç Bölge Maskesi (çizgiler hariç)")
            st.image(g["debug"]["adaptive_binary"], caption="Adaptif İkili (beyaz=zarar? ayarı değiştirilebilir)")
            st.image(g["debug"]["flake_interior"], caption="Kesik Hariç Zarar Maskesi")

st.caption("Kural: 0→C0, (0–1.25]→C1, (1.25–3.5]→C2, (3.5–8.5]→C3, (8.5–16.5]→C4, >16.5→C5. \
Hesap sadece **hücre içindeki** piksellerle yapılır; kesik çizgileri hariç tutulur.")
