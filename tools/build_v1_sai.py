import os
import cv2
import numpy as np

ROOT = r"C:\Users\hmind\.gemini\antigravity\scratch\cinematic-portfolio"
SRC_VIDEO = os.path.join(ROOT, "assets_src", "v1.mp4")
OUT_VIDEO = os.path.join(ROOT, "assets_src", "v1_sai.mp4")
TRACK_FILE = os.path.join(ROOT, "head_track.npz")
SAI_CUTOUT = os.path.join(ROOT, "sai_cutout.png")

def main():
    print("Loading track and assets...")
    track = np.load(TRACK_FILE)
    cx_arr = track['cx']
    cy_arr = track['cy']
    w_arr = track['w']

    sai_rgba = cv2.imread(SAI_CUTOUT, cv2.IMREAD_UNCHANGED)
    
    # Head crop with hair and neck
    hx, hy, hw, hh = 290, 290, 440, 520
    sai_head = sai_rgba[hy:hy+hh, hx:hx+hw]

    # Face center inside this crop
    face_cx_crop = (369 + 278 / 2.0) - hx
    face_cy_crop = (410 + 278 / 2.0) - hy

    cap = cv2.VideoCapture(SRC_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    print(f"Processing {total_frames} frames ({w}x{h} at {fps} fps)...")

    for i in range(total_frames):
        ok, fr = cap.read()
        if not ok:
            break

        cx = cx_arr[i]
        cy = cy_arr[i]
        fw = w_arr[i]

        scale = (fw / 278.0) * 1.08
        thw = int(hw * scale)
        thh = int(hh * scale)

        resized = cv2.resize(sai_head, (thw, thh), interpolation=cv2.INTER_LANCZOS4)

        top_left_x = int(cx - face_cx_crop * scale)
        top_left_y = int(cy - face_cy_crop * scale)

        # Feather mask
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        alpha = cv2.erode(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (9, 9), 0)

        # Soft neck blend into the collar
        for y in range(thh):
            if y > thh * 0.72:
                fade = (thh - y) / (thh * 0.28)
                alpha[y, :] *= max(0.0, min(1.0, fade))

        comp = fr.copy()

        x0 = max(0, top_left_x)
        y0 = max(0, top_left_y)
        x1 = min(w, top_left_x + thw)
        y1 = min(h, top_left_y + thh)

        sx0 = x0 - top_left_x
        sy0 = y0 - top_left_y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        if x1 > x0 and y1 > y0:
            a_sub = alpha[sy0:sy1, sx0:sx1, None]
            fg_sub = resized[sy0:sy1, sx0:sx1, :3]
            bg_sub = comp[y0:y1, x0:x1]
            comp[y0:y1, x0:x1] = (fg_sub * a_sub + bg_sub * (1.0 - a_sub)).astype(np.uint8)

        out.write(comp)

    cap.release()
    out.release()
    print("Done! Saved", OUT_VIDEO)

if __name__ == "__main__":
    main()
