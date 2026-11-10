import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter

ROOT = r"C:\Users\hmind\.gemini\antigravity\scratch\cinematic-portfolio"
OUT_DIR = os.path.join(ROOT, "public", "fin")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1600, 900
FONT_PATH = os.path.join(ROOT, "public", "fonts", "anton-400.woff2")
GRUNGE_PATH = os.path.join(ROOT, "public", "tex", "grunge.png")

def make_plate():
    print("Building plate with giant wordmark REDDY...")
    # Load original plate to keep the authentic red smoke & ember background lighting
    orig_plate = cv2.imread(os.path.join(OUT_DIR, "plate.jpg"))
    if orig_plate is None:
        orig_plate = np.zeros((H, W, 3), dtype=np.uint8)

    # Convert to PIL
    bg_pil = Image.fromarray(cv2.cvtColor(orig_plate, cv2.COLOR_BGR2RGB))
    
    # Inpaint/diffuse the old GIREESH text area in the background
    # GIREESH was mostly bright pixels (lum > 80)
    orig_gray = cv2.cvtColor(orig_plate, cv2.COLOR_BGR2GRAY)
    mask = (orig_gray > 75).astype(np.uint8) * 255
    # Expand slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Don't erase the deep red glow in lower half
    mask[750:, :] = 0
    mask[:80, :] = 0
    
    clean_bg = cv2.inpaint(orig_plate, mask, 21, cv2.INPAINT_TELEA)
    # Add deep red smoke gradient back
    for y in range(H):
        fy = y / float(H)
        red_boost = np.exp(-((fy - 0.65)**2) * 8.0) * 45.0
        clean_bg[y, :, 2] = np.clip(clean_bg[y, :, 2].astype(np.float32) + red_boost, 0, 255).astype(np.uint8)

    clean_bg_pil = Image.fromarray(cv2.cvtColor(clean_bg, cv2.COLOR_BGR2RGB))

    # Now render REDDY in Anton font
    text_canvas = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_canvas)

    try:
        font = ImageFont.truetype(FONT_PATH, 580)
    except Exception:
        font = ImageFont.truetype("arial.ttf", 520)

    text = "REDDY"
    # Measure
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Target: stretch to ~92% of frame width (approx 1470px)
    stretch_canvas = Image.new('RGBA', (tw + 20, th + 20), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(stretch_canvas)
    s_draw.text((-bbox[0] + 10, -bbox[1] + 10), text, font=font, fill=(240, 235, 230, 255))

    target_w = int(W * 0.92)
    target_h = int(H * 0.72)
    stretched = stretch_canvas.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Paste in center
    px = (W - target_w) // 2
    py = int(H * 0.15)
    text_canvas.paste(stretched, (px, py), stretched)

    # Texture the text with grunge
    grunge = Image.open(GRUNGE_PATH).convert('L').resize((W, H))
    g_arr = np.array(grunge, dtype=np.float32) / 255.0
    t_arr = np.array(text_canvas, dtype=np.float32)
    
    # Modulate text alpha and color with grunge
    wear = (g_arr * 0.35 + 0.65)
    for c in range(3):
        t_arr[:, :, c] *= wear
    t_arr[:, :, 3] *= (g_arr > 0.15).astype(np.float32)

    # Composite onto clean background
    alpha = t_arr[:, :, 3] / 255.0
    fg_rgb = t_arr[:, :, :3]
    bg_rgb = np.array(clean_bg_pil, dtype=np.float32)

    # Red atmospheric smoke overlap
    smoke_mask = np.exp(-((np.arange(H, dtype=np.float32) / float(H) - 0.58)**2) * 12.0)[:, None, None]
    smoke_col = np.array([190.0, 15.0, 10.0], dtype=np.float32)[None, None, :]
    
    comp_rgb = fg_rgb * alpha[..., None] + bg_rgb * (1.0 - alpha[..., None])
    # Add subtle red smoke over the letters
    comp_rgb = comp_rgb * (1.0 - smoke_mask * 0.28) + smoke_col * (smoke_mask * 0.28)

    # Darken vignette corners
    yy, xx = np.mgrid[0:H, 0:W]
    cx = W / 2.0; cy = H / 2.0
    dist = np.sqrt(((xx - cx) / cx)**2 + ((yy - cy) / cy)**2)
    vig = np.clip(1.0 - (dist - 0.55) * 0.9, 0.2, 1.0)[..., None]
    comp_rgb = np.clip(comp_rgb * vig, 0, 255).astype(np.uint8)

    # Convert back to BGR and save plate.jpg
    final_plate = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, "plate.jpg"), final_plate, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Saved plate.jpg with giant REDDY wordmark!")

def make_man():
    print("Extracting swapped man from Footer_image_swapped.jpg...")
    f1_swapped = cv2.imread(os.path.join(ROOT, "Footer_image_swapped.jpg"))
    if f1_swapped is None:
        print("Footer_image_swapped not found, using original man.png")
        return

    # Use the pipeline from extract_fin.py:
    # BOX = (580, 170, 1030, 900)
    x0, y0, x1, y1 = (580, 170, 1030, 900)
    rect = (x0, y0, x1 - x0, y1 - y0)
    gc = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(f1_swapped, gc, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)

    cv2.rectangle(gc, (815, 320), (890, 390), cv2.GC_FGD, -1)     # face
    cv2.line(gc, (865, 430), (900, 500), cv2.GC_FGD, 15)          # raised hand
    cv2.line(gc, (880, 400), (985, 400), cv2.GC_FGD, 5)           # cigarette
    cv2.line(gc, (780, 480), (780, 870), cv2.GC_FGD, 17)          # spine
    cv2.line(gc, (700, 300), (860, 230), cv2.GC_FGD, 9)           # hair mass
    cv2.line(gc, (990, 200), (1020, 320), cv2.GC_BGD, 11)
    cv2.line(gc, (620, 200), (640, 300), cv2.GC_BGD, 9)
    cv2.grabCut(f1_swapped, gc, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)

    region = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    gray = cv2.cvtColor(f1_swapped, cv2.COLOR_BGR2GRAY)
    core = np.zeros((H, W), np.uint8)
    core[y0:y1, x0:x1] = ((gray[y0:y1, x0:x1] < 78)).astype(np.uint8) * 255
    merged = np.maximum(region, core)

    n, lab, stats, _c = cv2.connectedComponentsWithStats(merged, 8)
    keep = lab[600, 800]
    if keep == 0:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = np.where(lab == keep, 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha[:y0, :] = 0
    alpha[:, :x0] = 0
    alpha[:, x1:] = 0

    pa = alpha[y0:, x0:x1]
    pc = f1_swapped[y0:, x0:x1]
    pc = cv2.addWeighted(pc, 1.30, cv2.GaussianBlur(pc, (0, 0), 1.1), -0.30, 0)
    cv2.imwrite(os.path.join(OUT_DIR, "man.png"), np.dstack([pc, pa]))
    print("Saved man.png with Sai's face and original rim lighting!")

if __name__ == "__main__":
    make_plate()
    make_man()
