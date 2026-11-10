import os
import time
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

ROOT = r"C:\Users\hmind\.gemini\antigravity\scratch\cinematic-portfolio"
SRC_VIDEO = os.path.join(ROOT, "assets_src", "v1.mp4")
OUT_VIDEO = os.path.join(ROOT, "assets_src", "v1_sai.mp4")
SAI_IMG_PATH = r"C:\Users\hmind\Downloads\sai image.jpeg"
TRACK_PATH = os.path.join(ROOT, "head_track.npz")

STEP = 4  # Keyframe every 4 frames (60 keyframes total across 240 frames)

def main():
    t0 = time.time()
    print("Loading track data...")
    track = np.load(TRACK_PATH)
    cx_arr = track['cx']
    cy_arr = track['cy']

    print("Extracting Sai face embedding...")
    app_full = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app_full.prepare(ctx_id=0, det_size=(640, 640))
    sai_img = cv2.imread(SAI_IMG_PATH)
    sai_faces = app_full.get(sai_img)
    if not sai_faces:
        raise RuntimeError("No face detected in Sai image!")
    sai_source = sai_faces[0]

    print("Loading detector and inswapper...")
    app_fast = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    app_fast.prepare(ctx_id=0, det_size=(256, 256))

    swapper_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(SRC_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames into RAM (240 frames of 720x1280 is ~600MB)
    print("Reading video frames into memory...")
    frames = []
    for _ in range(total_frames):
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()

    swapped_frames = [None] * len(frames)
    diff_patches = {}  # keyframe_idx -> (patch, x0, y0, x1, y1)

    pad = 140
    print(f"Computing AI face-swap on keyframes (every {STEP} frames)...")
    keyframe_indices = list(range(0, len(frames), STEP))
    if (len(frames) - 1) not in keyframe_indices:
        keyframe_indices.append(len(frames) - 1)

    for idx, k in enumerate(keyframe_indices):
        fr = frames[k]
        cx = int(cx_arr[k])
        cy = int(cy_arr[k])

        x0 = max(0, cx - pad)
        y0 = max(0, cy - pad)
        x1 = min(w, cx + pad)
        y1 = min(h, cy + pad)

        roi = fr[y0:y1, x0:x1]
        faces = app_fast.get(roi)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            face.bbox[0] += x0
            face.bbox[2] += x0
            face.bbox[1] += y0
            face.bbox[3] += y0
            face.kps[:, 0] += x0
            face.kps[:, 1] += y0
            swapped = swapper.get(fr, face, sai_source, paste_back=True)
        else:
            swapped = fr.copy()

        swapped_frames[k] = swapped

        # Store difference patch around face
        bx0 = max(0, cx - 110)
        by0 = max(0, cy - 120)
        bx1 = min(w, cx + 110)
        by1 = min(h, cy + 120)
        diff = swapped[by0:by1, bx0:bx1].copy()
        diff_patches[k] = (diff, bx0, by0, bx1, by1, cx, cy)

        if (idx + 1) % 10 == 0 or idx == len(keyframe_indices) - 1:
            print(f"Keyframe {idx+1}/{len(keyframe_indices)} done...", flush=True)

    print("Interpolating intermediate frames...")
    for i in range(len(frames)):
        if swapped_frames[i] is not None:
            continue
        # Find prev and next keyframe
        prev_k = (i // STEP) * STEP
        next_k = min(len(frames) - 1, prev_k + STEP)
        alpha_t = (i - prev_k) / float(max(1, next_k - prev_k))

        p_diff, p_x0, p_y0, p_x1, p_y1, p_cx, p_cy = diff_patches[prev_k]
        n_diff, n_x0, n_y0, n_x1, n_y1, n_cx, n_cy = diff_patches[next_k]

        cur_cx = cx_arr[i]
        cur_cy = cy_arr[i]

        # Target bounding box in current frame
        bx0 = int(cur_cx - 110)
        by0 = int(cur_cy - 120)
        bx1 = bx0 + (p_x1 - p_x0)
        by1 = by0 + (p_y1 - p_y0)

        # Blend patches
        blended_patch = cv2.addWeighted(p_diff, 1.0 - alpha_t, n_diff, alpha_t, 0)

        res = frames[i].copy()
        # Create elliptical feather mask
        pw = bx1 - bx0
        ph = by1 - by0
        mask = np.zeros((ph, pw), dtype=np.float32)
        cv2.ellipse(mask, (pw // 2, ph // 2), (pw // 2 - 8, ph // 2 - 8), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Paste back with feathering
        rx0 = max(0, bx0); ry0 = max(0, by0)
        rx1 = min(w, bx1); ry1 = min(h, by1)
        sx0 = rx0 - bx0; sy0 = ry0 - by0
        sx1 = sx0 + (rx1 - rx0); sy1 = sy0 + (ry1 - ry0)

        m_sub = mask[sy0:sy1, sx0:sx1, None]
        p_sub = blended_patch[sy0:sy1, sx0:sx1]
        b_sub = res[ry0:ry1, rx0:rx1]
        res[ry0:ry1, rx0:rx1] = (p_sub * m_sub + b_sub * (1.0 - m_sub)).astype(np.uint8)

        swapped_frames[i] = res

    print("Writing output video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
    for fr in swapped_frames:
        out.write(fr)
    out.release()

    dt = time.time() - t0
    print(f"SUCCESS! Output saved to {OUT_VIDEO} in {dt:.1f}s")

if __name__ == "__main__":
    main()
