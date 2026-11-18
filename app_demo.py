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
# 5. Giao diện 4 khu vực (cỡ lớn, dễ nhìn khi trình chiếu)
# ---------------------------------------------------------------------------
import inspect

H_IN, H_BIG, H_MID = 340, 520, 340   # chiều cao ô ảnh (px)

CSS = """
.gradio-container { max-width: 100% !important; font-size: 18px !important; }
.gradio-container, .gradio-container label, .gradio-container span,
.gradio-container input, .gradio-container p, .gradio-container li,
.gradio-container button, .gradio-container select { font-size: 18px !important; }
#app-title h1, #app-title h2 { font-size: 30px !important; }
.sec-title h2, .sec-title h3 { font-size: 24px !important; margin: 12px 0 4px 0 !important; }
#result-table table, #result-table th, #result-table td,
#result-table span, #result-table input { font-size: 20px !important; }
#result-table th { font-weight: 700 !important; }
#run-btn { font-size: 26px !important; font-weight: 700 !important; min-height: 70px !important; }
#note-line p { font-size: 18px !important; }

/* Màu chữ tương phản cao (sáng: gần đen, tối: gần trắng) */
:root, .gradio-container {
  --body-text-color: #0f172a !important;
  --body-text-color-subdued: #334155 !important;
  --block-label-text-color: #0f172a !important;
  --block-title-text-color: #0f172a !important;
  --block-info-text-color: #334155 !important;
  --input-text-color: #0f172a !important;
  --table-text-color: #0f172a !important;
}
.dark, .dark .gradio-container {
  --body-text-color: #f8fafc !important;
  --body-text-color-subdued: #cbd5e1 !important;
  --block-label-text-color: #f8fafc !important;
  --block-title-text-color: #f8fafc !important;
  --block-info-text-color: #cbd5e1 !important;
  --input-text-color: #f8fafc !important;
  --table-text-color: #f8fafc !important;
}
.gradio-container label, .gradio-container label span { font-weight: 600 !important; }

/* Ô chọn trọng số cỡ lớn */
#weight-dd label span, #weight-dd [data-testid="block-info"] { font-size: 22px !important; font-weight: 700 !important; }
#weight-dd input { font-size: 22px !important; font-weight: 600 !important; min-height: 56px !important; }
#weight-dd .wrap, #weight-dd .wrap-inner { min-height: 60px !important; }
ul.options li, ul.options li span { font-size: 20px !important; padding: 10px 14px !important; }
"""

# Gradio 6: css ở launch(); Gradio 4/5: css ở gr.Blocks(). fill_width chỉ dùng nếu có.
_blocks_params = inspect.signature(gr.Blocks.__init__).parameters
_launch_params = inspect.signature(gr.Blocks.launch).parameters
blocks_kw, launch_kw = {"title": "HVI-CIDNet Demo"}, {}
if "fill_width" in _blocks_params:
    blocks_kw["fill_width"] = True
if "css" in _launch_params:
    launch_kw["css"] = CSS
else:
    blocks_kw["css"] = CSS


def _out(label, height):
    return gr.Image(label=label, type="numpy", height=height, interactive=False)


with gr.Blocks(**blocks_kw) as demo:
    gr.Markdown("## HVI: A New Color Space for Low-light Image Enhancement (CVPR 2025) — Demo nhóm",
                elem_id="app-title")

    # Khu vực ①: Ảnh vào + tham số
    gr.Markdown("### ① Ảnh vào & tham số", elem_classes="sec-title")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Row():
                inp = gr.Image(label="Ảnh thiếu sáng", type="numpy", height=H_IN)
                gt = gr.Image(label="Ảnh chuẩn GT (tùy chọn, để tính PSNR/SSIM)", type="numpy",
                              height=H_IN)
        with gr.Column(scale=2):
            w = gr.Dropdown(WEIGHTS, value=WEIGHTS[0], allow_custom_value=True,
                            label="Trọng số (.pth trong weights/ hoặc mã Hugging Face)",
                            elem_id="weight-dd")
            a_i = gr.Slider(0.1, 2.0, 1.0, step=0.05, label="α_I — độ sáng (nhân vào RGB đầu ra PHVIT)")
            a_s = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="α_S — độ bão hòa (nhân vào S trong PHVIT)")
            gamma = gr.Slider(0.1, 5.0, 1.0, step=0.05, label="gamma (lũy thừa ảnh đầu vào)")
            use_k = gr.Checkbox(True, label="Dùng k đã học (bỏ tick để tự đặt k)")
            k = gr.Slider(0.5, 50, 5.0, step=0.5, label="k (ký hiệu bài báo; code lưu density_k = 1/k)")
            g_cls = gr.Slider(0.1, 1.0, 0.4, step=0.05, label="gamma cho Gamma Correction cổ điển")
            gr.Markdown("Gợi ý theo eval.py của repo: LOL-v1 → α_S = 1.3, α_I = 1.0; "
                        "LOL-v2-Real → α_S = 1.0, α_I = 0.80–0.84. App gốc mặc định cả hai = 1.0.")
            btn = gr.Button("Xử lý", variant="primary", size="lg", elem_id="run-btn")

    # Khu vực ②: Ảnh gốc vs HVI-CIDNet + bảng chỉ số
    gr.Markdown("### ② Ảnh gốc vs HVI-CIDNet", elem_classes="sec-title")
    with gr.Row():
        orig = _out("Ảnh gốc (thiếu sáng)", H_BIG)
        o4 = _out("HVI-CIDNet", H_BIG)
    table = gr.Dataframe(headers=["Phương pháp", "PSNR ↑", "SSIM ↑", "NIQE ↓", "Thời gian (ms)"],
                         interactive=False, wrap=True, elem_id="result-table")
    note = gr.Markdown(elem_id="note-line")

    # Khu vực ③: 4 phương pháp cổ điển (lưới 2×2)
    gr.Markdown("### ③ Phương pháp cổ điển", elem_classes="sec-title")
    with gr.Row():
        o1 = _out("Gamma (tự code)", H_MID)
        o5 = _out("HE (tự code)", H_MID)
    with gr.Row():
        o2 = _out("CLAHE (tự code)", H_MID)
        o3 = _out("CLAHE (OpenCV)", H_MID)

    # Khu vực ④: Kênh màu HSV vs HVI
    gr.Markdown("### ④ Kênh màu của ảnh vào: HSV (trên) vs HVI (dưới)", elem_classes="sec-title")
    with gr.Row():
        h1 = _out("H (HSV)", H_MID); s1 = _out("S (HSV)", H_MID); v1 = _out("V (HSV)", H_MID)
    with gr.Row():
        h2 = _out("Ĥ (HVI)", H_MID); v2 = _out("V̂ (HVI)", H_MID); i2 = _out("I_max (HVI)", H_MID)

    btn.click(process, [inp, gt, w, gamma, a_i, a_s, k, use_k, g_cls],
              [o1, o5, o2, o3, o4, h1, s1, v1, h2, v2, i2, table, note]
              ).then(lambda x: x, inp, orig)

if __name__ == "__main__":
    demo.launch(server_port=args.port, **launch_kw)
    