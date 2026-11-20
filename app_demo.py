"""
app_demo.py — Web Demo HVI-CIDNet (Thành viên 4)
Đặt file này ở THƯ MỤC GỐC repo HVI-CIDNet (cùng cấp với thư mục net/, weights/).
Chạy:  python app_demo.py            (GPU nếu có)
       python app_demo.py --cpu      (ép chạy CPU)
Cần:   pip install torch torchvision gradio opencv-python scikit-image pillow numpy scipy
                   einops huggingface_hub safetensors matplotlib
Cùng thư mục cần có: color_space_scratch.py (TV1)
"""
import argparse, glob, os, time

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from net.CIDNet import CIDNet

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()
DEVICE = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"

# ---------------------------------------------------------------------------
# 1. Thuật toán cổ điển TỰ CODE — dùng module của Thành viên 1 (color_space_scratch.py).
#    TV1 cung cấp hàm chuẩn cho demo: enhance_classical_scratch(img_rgb, method, gamma)
#    với method = "gamma" | "he" | "clahe" (CLAHE: clip_limit=2.0, lưới 8x8, trên kênh V của HSV).
#    Input/Output: ảnh RGB uint8 HxWx3.
# ---------------------------------------------------------------------------
try:
    from color_space_scratch import enhance_classical_scratch, srgb_to_hsv_scratch
    SCRATCH_SRC = "color_space_scratch.py (TV1)"
except ImportError:
    SCRATCH_SRC = "BẢN TẠM trong app_demo.py — chưa chép file color_space_scratch.py của TV1 vào!"
    srgb_to_hsv_scratch = None

    def enhance_classical_scratch(img, method="clahe", gamma=0.4):
        if method == "gamma":
            lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).round().astype(np.uint8)
            return lut[img]
        v = img.max(axis=2)  # HE toàn cục trên kênh V, giữ tỉ lệ màu (dùng tạm cho cả "he"/"clahe")
        cdf = np.bincount(v.ravel(), minlength=256).cumsum() / v.size
        ratio = (cdf[v] * 255.0) / (v.astype(np.float32) + 1e-6)
        return np.clip(img * ratio[..., None], 0, 255).astype(np.uint8)


def opencv_clahe(img, clip=2.0, tile=8):
    """CLAHE của OpenCV, cùng tham số và cùng kênh (V của HSV) với clahe_scratch của TV1."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# 2. Mô hình HVI-CIDNet
# ---------------------------------------------------------------------------
# Nguồn trọng số: (a) file .pth tải từ Baidu Pan / One Drive trong README, đặt vào weights/
#                 (b) mã Hugging Face, ví dụ "Fediory/HVI-CIDNet-LOLv1-wperc" (giống eval_hf.py)
HF_DEFAULT = "Fediory/HVI-CIDNet-LOLv1-wperc"
WEIGHTS = sorted(p for p in glob.glob("weights/**/*.pth", recursive=True)
                 if "train" not in p.split(os.sep)) + [HF_DEFAULT]
model = CIDNet().to(DEVICE)
_loaded = {"path": None, "k_learned": None}


def load_weights(path):
    if _loaded["path"] != path:
        if path.endswith(".pth"):
            state = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state)
        else:  # mã Hugging Face — cách nạp y hệt hàm from_pretrained trong eval_hf.py
            from huggingface_hub import hf_hub_download
            import safetensors.torch as sf
            f = hf_hub_download(repo_id=path, filename="model.safetensors", repo_type="model")
            model.load_state_dict(sf.load_file(f), strict=False)
        model.eval()
        _loaded["path"] = path
        # density_k trong code = 1/k của bài báo (comment trong HVI_transform.py)
        _loaded["k_learned"] = float(model.trans.density_k.item())
    return 1.0 / _loaded["k_learned"]  # trả về k theo ký hiệu bài báo


def to_tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)


def to_uint8(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)


def reflect_pad8(x):
    """Đệm Reflect ở cạnh PHẢI và DƯỚI để H, W chia hết cho 8 (3 tầng downsample)."""
    h, w = x.shape[2], x.shape[3]
    padh = (8 - h % 8) % 8
    padw = (8 - w % 8) % 8
    return F.pad(x, (0, padw, 0, padh), mode="reflect"), h, w


@torch.no_grad()
def run_cidnet(img, weight_path, gamma, alpha_i, alpha_s, k_paper, use_learned_k):
    k_learned = load_weights(weight_path)
    # Ghi đè k (để minh họa Failure Case 2) hoặc giữ k đã học
    k_use = k_learned if use_learned_k else k_paper
    model.trans.density_k.data.fill_(1.0 / k_use)

    model.trans.gated = True      # bật nhân alpha_s vào độ bão hòa S trong PHVIT
    model.trans.gated2 = True     # bật nhân alpha (alpha_i) vào ảnh RGB đầu ra của PHVIT
    model.trans.alpha_s = alpha_s
    model.trans.alpha = alpha_i

    x = to_tensor(img)
    x, h, w = reflect_pad8(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    y = model(x ** gamma)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    y = torch.clamp(y, 0, 1)[:, :, :h, :w]       # Clip Operation + cắt về kích thước gốc

    hvi = model.trans.HVIT(to_tensor(img))        # kênh HVI của ẢNH VÀO (với k đang dùng)
    model.trans.density_k.data.fill_(1.0 / k_learned)  # trả lại k gốc
    return to_uint8(y), hvi[0].cpu().numpy(), dt, k_use


# ---------------------------------------------------------------------------
# 3. Trực quan hóa kênh màu HSV vs HVI
# ---------------------------------------------------------------------------
def gray(ch, lo, hi):
    ch = (np.clip(ch, lo, hi) - lo) / (hi - lo + 1e-8)
    return (ch * 255).astype(np.uint8)


def hsv_channels(img):
    if srgb_to_hsv_scratch is not None:  # dùng đúng hàm HSV tự code của TV1 (h trong [0, 6])
        hsv = srgb_to_hsv_scratch(img.astype(np.float64) / 255.0)
        return [gray(hsv[..., 0], 0, 6), gray(hsv[..., 1], 0, 1), gray(hsv[..., 2], 0, 1)]
    rgb = img.astype(np.float32) / 255.0
    mx, mn = rgb.max(2), rgb.min(2)
    d = mx - mn
    s = np.where(mx > 0, d / (mx + 1e-8), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    m = d > 0
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    return [gray(h, 0, 6), gray(s, 0, 1), gray(mx, 0, 1)]


# ---------------------------------------------------------------------------
# 4. Hàm xử lý chính
# ---------------------------------------------------------------------------
try:  # NIQE dùng đúng hàm của repo (loss/niqe_utils.py) — không cần ảnh GT
    from loss.niqe_utils import calculate_niqe
except Exception:
    calculate_niqe = None


def metrics(out, gt):
    niqe = f"{calculate_niqe(out):.3f}" if calculate_niqe else "—"
    if gt is None:
        return "—", "—", niqe
    if gt.shape != out.shape:
        gt = cv2.resize(gt, (out.shape[1], out.shape[0]))
    p = peak_signal_noise_ratio(gt, out, data_range=255)
    s = structural_similarity(gt, out, channel_axis=2, data_range=255)
    return f"{p:.3f}", f"{s:.4f}", niqe


def process(img, gt, weight_path, gamma, alpha_i, alpha_s, k_paper, use_learned_k, gamma_classic):
    if img is None:
        raise gr.Error("Hãy tải ảnh thiếu sáng lên.")
    if not weight_path:
        raise gr.Error("Chọn file .pth trong weights/ hoặc nhập mã Hugging Face.")
    img = np.array(Image.fromarray(img).convert("RGB"))
    gt = None if gt is None else np.array(Image.fromarray(gt).convert("RGB"))

    rows, outs = [], {}
    jobs = [("Gamma (tự code)", lambda: enhance_classical_scratch(img, "gamma", gamma_classic)),
            ("HE (tự code)", lambda: enhance_classical_scratch(img, "he")),
            ("CLAHE (tự code)", lambda: enhance_classical_scratch(img, "clahe")),
            ("CLAHE (OpenCV)", lambda: opencv_clahe(img))]
    for name, fn in jobs:
        t0 = time.time(); outs[name] = fn(); t1 = time.time() - t0
        rows.append([name, *metrics(outs[name], gt), f"{t1*1000:.1f}"])

    out, hvi, dt, k_use = run_cidnet(img, weight_path, gamma, alpha_i, alpha_s, k_paper, use_learned_k)
    outs["HVI-CIDNet"] = out
    rows.append([f"HVI-CIDNet (k={k_use:.2f})", *metrics(out, gt), f"{dt*1000:.1f}"])

    hsv = hsv_channels(img)
    hvi_imgs = [gray(hvi[0], -1, 1), gray(hvi[1], -1, 1), gray(hvi[2], 0, 1)]
    note = f"Thuật toán tự code lấy từ: {SCRATCH_SRC} | Thiết bị: {DEVICE}"
    return (outs["Gamma (tự code)"], outs["HE (tự code)"], outs["CLAHE (tự code)"], outs["CLAHE (OpenCV)"], out,
            *hsv, *hvi_imgs, rows, note)


# ---------------------------------------------------------------------------
# 5. Giao diện 4 khu vực (thiết kế cho trình chiếu)
# ---------------------------------------------------------------------------
import inspect

H_IN, H_BIG, H_MID = 340, 520, 340   # chiều cao ô ảnh (px)

# Bộ tham số gợi ý: chỉ đặt giá trị cho các ô điều khiển, không đổi logic xử lý.
# Thứ tự giá trị: (trọng số, α_I, α_S, gamma, dùng k đã học)
PRESETS = {
    "Mặc định": (WEIGHTS[0], 1.0, 1.0, 1.0, True),
    "LOL-v1": ("Fediory/HVI-CIDNet-LOLv1-wperc", 1.0, 1.3, 1.0, True),
    "LOL-v2-Real": ("Fediory/HVI-CIDNet-LOLv2-real-bestSSIM", 0.82, 1.0, 1.0, True),
    "Ảnh tự chụp (TV3)": ("Fediory/HVI-CIDNet-LOLv2-syn-wperc", 1.0, 1.0, 1.0, True),
}

# Ảnh mẫu bấm-là-chạy (bộ 20 ảnh tự chụp của TV3 nếu có trong test_tv3/input)
_EX_NAMES = ["red_01", "night_01", "backlit_01", "dark_01", "red_02", "night_05"]
EXAMPLES = [[p] for p in (os.path.join("test_tv3", "input", n + ".png") for n in _EX_NAMES)
            if os.path.exists(p)]

try:
    THEME = gr.themes.Soft(
        primary_hue="indigo", secondary_hue="amber", neutral_hue="slate",
        text_size=gr.themes.sizes.text_lg, radius_size=gr.themes.sizes.radius_lg,
        # Inter: đủ dấu tiếng Việt + chữ Hy Lạp (α, γ, ε) + dấu mũ (Ĥ, V̂) nên ký hiệu không bị lẫn phông
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Arial", "sans-serif"])
except Exception:
    THEME = None

CSS = """
.gradio-container { max-width: 100% !important; padding: 0 28px 32px !important; font-size: 18px !important; }
.gradio-container label, .gradio-container input, .gradio-container p, .gradio-container li,
.gradio-container button, .gradio-container select { font-size: 18px !important; }

/* Màu chữ tương phản cao (sáng: gần đen, tối: gần trắng) */
:root, .gradio-container {
  --body-text-color: #0f172a !important; --body-text-color-subdued: #334155 !important;
  --block-label-text-color: #0f172a !important; --block-title-text-color: #0f172a !important;
  --block-info-text-color: #334155 !important; --input-text-color: #0f172a !important;
  --table-text-color: #0f172a !important;
}
.dark, .dark .gradio-container {
  --body-text-color: #f8fafc !important; --body-text-color-subdued: #cbd5e1 !important;
  --block-label-text-color: #f8fafc !important; --block-title-text-color: #f8fafc !important;
  --block-info-text-color: #cbd5e1 !important; --input-text-color: #f8fafc !important;
  --table-text-color: #f8fafc !important;
}
.gradio-container label, .gradio-container label span { font-weight: 600 !important; }
.gradio-container { line-height: 1.55; font-feature-settings: "tnum" 1; }
/* Bỏ nền xanh của nhãn cho đỡ rối mắt */
:root, .dark { --block-label-background-fill: transparent !important; --block-title-background-fill: transparent !important;
               --block-label-border-width: 0px !important; }
[data-testid="block-label"], [data-testid="block-info"] { box-shadow: none !important; }

/* Nhãn tham số có ký hiệu toán học chuẩn (chỉ số dưới) */
.plabel { display: flex; align-items: baseline; gap: 10px; margin: 4px 0 0; }
.plabel .sym { font-family: "Cambria Math", "Times New Roman", serif; font-style: italic; font-size: 24px !important;
               font-weight: 600; color: #4338ca; min-width: 34px; }
.dark .plabel .sym { color: #a5b4fc; }
.plabel .sym sub { font-size: 15px !important; font-style: italic; }
.plabel .pname { font-size: 18px !important; font-weight: 700; }
.pdesc { font-size: 15px !important; opacity: .75; margin: 0 0 -22px 44px; }
.pdesc code, .formula code { font-size: 14px !important; background: var(--color-accent-soft); padding: 1px 6px; border-radius: 6px; }
.formula { background: var(--color-accent-soft); border-radius: 14px; padding: 10px 16px !important; }
.formula p { font-size: 17px !important; margin: 4px 0 !important; }
.formula .katex { font-size: 1.12em !important; }
.hint p { font-size: 16px !important; opacity: .85; }
.hint .katex { font-size: 1.05em !important; }

/* Banner tiêu đề */
#hero { background: linear-gradient(120deg, #1e1b4b 0%, #3730a3 50%, #6d28d9 100%);
        border-radius: 20px; padding: 26px 34px; margin: 18px 0 6px; box-shadow: 0 10px 30px rgba(49,46,129,.25); }
#hero h1 { color: #fff !important; font-size: 34px !important; font-weight: 800 !important; margin: 10px 0 6px !important; }
#hero p { color: #e0e7ff !important; font-size: 19px !important; margin: 0 !important; }
#hero .badge { display: inline-block; background: rgba(255,255,255,.16); color: #fff !important;
               border: 1px solid rgba(255,255,255,.35); border-radius: 999px; padding: 4px 14px;
               margin-right: 8px; font-size: 15px !important; font-weight: 600; }
#hero .badge.hl { background: #fbbf24; color: #1e1b4b !important; border-color: #fbbf24; }

/* Thẻ cho từng khu vực */
.card { background: var(--block-background-fill) !important; border: 1px solid var(--border-color-primary) !important;
        border-radius: 20px !important; padding: 18px 20px !important; box-shadow: 0 6px 22px rgba(15,23,42,.07) !important; }
.sec-head { display: flex; align-items: center; gap: 14px; margin: 2px 0 6px; }
.sec-num { flex: none; width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
           background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #fff !important;
           font-size: 22px !important; font-weight: 800; }
.sec-title { font-size: 24px !important; font-weight: 800; line-height: 1.2; }
.sec-sub { font-size: 16px !important; opacity: .75; margin-top: 2px; }
.sub-label { font-size: 16px !important; font-weight: 700; opacity: .8; margin: 6px 0 -4px; }

/* Ô chọn trọng số cỡ lớn */
#weight-dd label span, #weight-dd [data-testid="block-info"] { font-size: 22px !important; font-weight: 700 !important; }
#weight-dd input { font-size: 21px !important; font-weight: 600 !important; min-height: 54px !important; }
#weight-dd .wrap, #weight-dd .wrap-inner { min-height: 58px !important; }
ul.options li, ul.options li span { font-size: 20px !important; padding: 10px 14px !important; }

/* Nút */
.preset-btn button, button.preset-btn { font-size: 16px !important; font-weight: 600 !important; }
#run-btn { font-size: 26px !important; font-weight: 800 !important; min-height: 72px !important; letter-spacing: .5px;
           background: linear-gradient(120deg, #4f46e5, #7c3aed) !important; color: #fff !important;
           border: none !important; box-shadow: 0 8px 22px rgba(79,70,229,.35) !important; }
#run-btn:hover { filter: brightness(1.08); }
#run-btn:disabled { cursor: progress !important; opacity: .9 !important; animation: runpulse 1.1s ease-in-out infinite; }
@keyframes runpulse { 0%, 100% { filter: brightness(1); } 50% { filter: brightness(1.3); } }
#run-status .st { padding: 10px 16px; border-radius: 12px; font-size: 18px !important; font-weight: 600; }
#run-status .st.idle { background: var(--color-accent-soft); }
#run-status .st.busy { background: #fef3c7; color: #92400e !important; }
#run-status .st.ok { background: #dcfce7; color: #166534 !important; }
#run-status .st.err { background: #fee2e2; color: #991b1b !important; }
#run-status .spin { display: inline-block; width: 18px; height: 18px; margin-right: 10px; vertical-align: -3px;
                    border: 3px solid currentColor; border-right-color: transparent; border-radius: 50%;
                    animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Bảng kết quả */
#result-table table, #result-table th, #result-table td, #result-table span, #result-table input { font-size: 20px !important; }
#result-table th { font-weight: 800 !important; }
#result-table, #result-table * { font-family: var(--font) !important; }
.in-row { flex-grow: 0 !important; align-items: flex-start !important; }
.card { justify-content: flex-start !important; }
.howto { background: var(--color-accent-soft); border-radius: 14px; padding: 12px 18px; margin-top: 4px; }
.howto b { font-size: 18px !important; }
.howto ol { margin: 6px 0 0 20px !important; padding: 0 !important; }
.howto li { font-size: 17px !important; margin: 3px 0 !important; }
.gallery img, .gallery-item img, button.gallery-item { min-width: 96px; min-height: 96px; }
.card > * { flex-grow: 0 !important; }
#result-table tbody tr:last-child td { font-weight: 800 !important; background: var(--color-accent-soft) !important; }
#note-line { background: var(--color-accent-soft); border-left: 5px solid #6366f1; border-radius: 10px; padding: 8px 16px; }
#note-line p { font-size: 17px !important; margin: 0 !important; }
#footer { text-align: center; opacity: .7; font-size: 15px !important; margin-top: 10px; }
"""

# Gradio 6: css/theme ở launch(); Gradio 4/5: ở gr.Blocks(). fill_width chỉ dùng nếu có.
_blocks_params = inspect.signature(gr.Blocks.__init__).parameters
_launch_params = inspect.signature(gr.Blocks.launch).parameters
blocks_kw, launch_kw = {"title": "HVI-CIDNet Demo"}, {}
if "fill_width" in _blocks_params:
    blocks_kw["fill_width"] = True
for _name, _val in (("css", CSS), ("theme", THEME)):
    if _val is None:
        continue
    if _name in _launch_params:
        launch_kw[_name] = _val
    elif _name in _blocks_params:
        blocks_kw[_name] = _val


def _out(label, height):
    return gr.Image(label=label, type="numpy", height=height, interactive=False)


RUN_LABEL, BUSY_LABEL = "▶  Xử lý", "⏳  Đang xử lý…"


def _status(kind, text):
    spin = '<span class="spin"></span>' if kind == "busy" else ""
    return f'<div class="st {kind}">{spin}{text}</div>'


def _on_start():
    return (gr.update(value=BUSY_LABEL, interactive=False),
            _status("busy", f"Đang chạy 4 thuật toán cổ điển + HVI-CIDNet trên {DEVICE.upper()}… "
                            "(CPU mất khoảng 3–5 giây, lần đầu đổi trọng số lâu hơn vì phải tải)"),
            "", time.time())


def _on_done(note_text, t0):
    dt = time.time() - t0 if t0 else 0.0
    if note_text:  # process() trả về dòng ghi chú khi chạy thành công
        msg = _status("ok", f"✅ Xong sau {dt:.1f} giây — kéo xuống xem khu vực ② ③ ④")
    else:
        msg = _status("err", "⚠ Chưa xử lý được — xem thông báo lỗi ở góc màn hình (thiếu ảnh hoặc trọng số?)")
    return gr.update(value=RUN_LABEL, interactive=True), msg


MATH = [{"left": "$", "right": "$", "display": False}]


def _param(sym, name, desc):
    """Nhãn hiển thị cho thanh trượt: ký hiệu đúng chuẩn (HTML sub) + tên + mô tả ngắn."""
    gr.HTML(f'<div class="plabel"><span class="sym">{sym}</span><span class="pname">{name}</span></div>'
            f'<div class="pdesc">{desc}</div>')


def _section(num, title, sub=""):
    gr.HTML(f'<div class="sec-head"><div class="sec-num">{num}</div><div>'
            f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div></div></div>')


with gr.Blocks(**blocks_kw) as demo:
    gr.HTML(f"""<div id="hero">
      <span class="badge hl">CVPR 2025</span><span class="badge">Low-light Image Enhancement</span>
      <span class="badge">Thiết bị: {DEVICE.upper()}</span>
      <h1>HVI-CIDNet — Tăng cường ảnh thiếu sáng</h1>
      <p>HVI: A New Color Space for Low-light Image Enhancement · Demo nhóm ·
         So sánh thuật toán cổ điển tự code, OpenCV và HVI-CIDNet</p></div>""")

    # Khu vực ①: Ảnh vào + tham số
    with gr.Row(equal_height=True):
        with gr.Column(scale=5, elem_classes="card"):
            _section("1", "Ảnh vào", "Tải ảnh thiếu sáng; ảnh GT chỉ cần khi muốn tính PSNR/SSIM")
            with gr.Row(elem_classes="in-row"):
                inp = gr.Image(label="Ảnh thiếu sáng", type="numpy", height=H_IN)
                gt = gr.Image(label="Ảnh chuẩn GT (tùy chọn, để tính PSNR/SSIM)", type="numpy",
                              height=H_IN)
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=[inp], label="Ảnh mẫu (bộ ảnh tự chụp của TV3) — bấm để chọn",
                            examples_per_page=6)
            gr.HTML("""<div class="howto"><b>Cách dùng nhanh</b><ol>
              <li>Chọn ảnh mẫu ở trên hoặc tải ảnh thiếu sáng của bạn (thêm ảnh GT nếu muốn có PSNR/SSIM).</li>
              <li>Bấm một <b>Preset</b> hoặc tự kéo <i>α<sub>I</sub></i>, <i>α<sub>S</sub></i>, <i>k</i> ở cột Tham số.</li>
              <li>Bấm <b>▶ Xử lý</b> — kết quả hiện ở khu vực ②, ③, ④ bên dưới.</li></ol></div>""")
            # Công thức các thanh trượt tác động (khớp net/HVI_transform.py: HVIT/PHVIT, gated/gated2)
            gr.Markdown(
                r"**Các thanh trượt tác động ở đâu** (khớp `net/HVI_transform.py`)" "\n\n"
                r"• Hàm nén vùng tối: $C_k = \big(\sin(\pi I_{max}/2) + \varepsilon\big)^{1/k}$" "\n\n"
                r"• Độ bão hòa: $S = \alpha_S \sqrt{\hat h^{2} + \hat v^{2}}$ , với "
                r"$\hat h = \hat H / (C_k + \varepsilon)$ , $\hat v = \hat V / (C_k + \varepsilon)$" "\n\n"
                r"• Độ sáng: $\mathrm{RGB}_{out} = \alpha_I \cdot \mathrm{RGB}_{\mathrm{PHVIT}}$",
                latex_delimiters=MATH, elem_classes="formula")
        with gr.Column(scale=4, elem_classes="card"):
            _section("⚙", "Tham số", "Kéo thanh trượt rồi bấm Xử lý")
            w = gr.Dropdown(WEIGHTS, value=WEIGHTS[0], allow_custom_value=True,
                            label="Trọng số (.pth trong weights/ hoặc mã Hugging Face)",
                            elem_id="weight-dd")
            # Thanh trượt giữ nguyên khoảng/mặc định/nhãn; nhãn hiển thị thay bằng HTML có chỉ số dưới
            with gr.Row():
                with gr.Column(min_width=240):
                    _param("α<sub>I</sub>", "Độ sáng", "nhân vào RGB đầu ra PHVIT")
                    a_i = gr.Slider(0.1, 2.0, 1.0, step=0.05, show_label=False,
                                    label="α_I — độ sáng (nhân vào RGB đầu ra PHVIT)")
                with gr.Column(min_width=240):
                    _param("α<sub>S</sub>", "Độ bão hòa", "nhân vào S trong PHVIT")
                    a_s = gr.Slider(0.0, 2.0, 1.0, step=0.05, show_label=False,
                                    label="α_S — độ bão hòa (nhân vào S trong PHVIT)")
            with gr.Row():
                with gr.Column(min_width=240):
                    _param("γ", "Gamma ảnh vào", "mạng nhận x<sup>γ</sup> (lũy thừa ảnh đầu vào)")
                    gamma = gr.Slider(0.1, 5.0, 1.0, step=0.05, show_label=False,
                                      label="gamma (lũy thừa ảnh đầu vào)")
                with gr.Column(min_width=240):
                    _param("γ<sub>c</sub>", "Gamma cổ điển", "cho Gamma Correction cổ điển (tự code)")
                    g_cls = gr.Slider(0.1, 1.0, 0.4, step=0.05, show_label=False,
                                      label="gamma cho Gamma Correction cổ điển")
            _param("k", "Mật độ vùng tối", "ký hiệu bài báo; code lưu <code>density_k = 1/k</code>")
            with gr.Row(equal_height=True):
                use_k = gr.Checkbox(True, label="Dùng k đã học (bỏ tick để tự đặt k)", scale=1)
                k = gr.Slider(0.5, 50, 5.0, step=0.5, show_label=False,
                              label="k (ký hiệu bài báo; code lưu density_k = 1/k)", scale=2)
            gr.HTML('<div class="sub-label">Preset nhanh</div>')
            with gr.Row():
                preset_btns = {name: gr.Button(name, size="sm", variant="secondary", elem_classes="preset-btn")
                               for name in PRESETS}
            gr.Markdown(r"Gợi ý theo `eval.py` của repo: LOL-v1 → $\alpha_S = 1.3,\ \alpha_I = 1.0$ ; "
                        r"LOL-v2-Real → $\alpha_S = 1.0,\ \alpha_I = 0.80\text{–}0.84$. "
                        r"App gốc mặc định cả hai $= 1.0$.", latex_delimiters=MATH, elem_classes="hint")
            btn = gr.Button(RUN_LABEL, variant="primary", size="lg", elem_id="run-btn")
            status = gr.HTML(_status("idle", "Sẵn sàng — chọn ảnh rồi bấm Xử lý"), elem_id="run-status")
            t_start = gr.State(0.0)

    # Khu vực ②: Ảnh gốc vs HVI-CIDNet + bảng chỉ số
    with gr.Column(elem_classes="card"):
        _section("2", "Ảnh gốc vs HVI-CIDNet", "Kết quả chính và bảng chỉ số của cả 5 phương pháp")
        with gr.Row():
            orig = _out("Ảnh gốc (thiếu sáng)", H_BIG)
            o4 = _out("HVI-CIDNet", H_BIG)
        table = gr.Dataframe(headers=["Phương pháp", "PSNR ↑", "SSIM ↑", "NIQE ↓", "Thời gian (ms)"],
                             interactive=False, wrap=True, elem_id="result-table")
        note = gr.Markdown(elem_id="note-line")

    # Khu vực ③ (lưới 2×2) và ④ (kênh màu 2×3) đặt cạnh nhau, cùng 2 hàng nên cao bằng nhau
    with gr.Row(equal_height=True):
        with gr.Column(scale=2, elem_classes="card"):
            _section("3", "Phương pháp cổ điển", "Gamma, HE, CLAHE tự code (TV1) và CLAHE OpenCV")
            with gr.Row():
                o1 = _out("Gamma (tự code)", H_MID)
                o5 = _out("HE (tự code)", H_MID)
            with gr.Row():
                o2 = _out("CLAHE (tự code)", H_MID)
                o3 = _out("CLAHE (OpenCV)", H_MID)
        with gr.Column(scale=3, elem_classes="card"):
            _section("4", "Kênh màu của ảnh vào",
                     "HSV: <i>H, S, V</i> (hàng trên) · HVI: <i>Ĥ, V̂, I<sub>max</sub></i> (hàng dưới)")
            with gr.Row():
                h1 = _out("H (HSV)", H_MID); s1 = _out("S (HSV)", H_MID); v1 = _out("V (HSV)", H_MID)
            with gr.Row():
                h2 = _out("Ĥ (HVI)", H_MID); v2 = _out("V̂ (HVI)", H_MID); i2 = _out("Iₘₐₓ (HVI)", H_MID)

    gr.HTML('<div id="footer">Thành viên 4 · Tích hợp pipeline (Reflect Padding 8×8, Clip Operation, '
            'PHVIT α_I / α_S / k) &amp; Web Demo</div>')

    for _name, _b in preset_btns.items():
        _b.click(lambda v=PRESETS[_name]: v, None, [w, a_i, a_s, gamma, use_k])

    # Phản hồi ngay khi bấm: khóa nút + báo "đang xử lý" (listener phụ, chạy tức thì, không qua hàng đợi)
    btn.click(_on_start, None, [btn, status, note, t_start], queue=False, show_progress="hidden")

    btn.click(process, [inp, gt, w, gamma, a_i, a_s, k, use_k, g_cls],
              [o1, o5, o2, o3, o4, h1, s1, v1, h2, v2, i2, table, note]
              ).then(lambda x: x, inp, orig
              ).then(_on_done, [note, t_start], [btn, status], show_progress="hidden")

if __name__ == "__main__":
    demo.launch(server_port=args.port, **launch_kw)

