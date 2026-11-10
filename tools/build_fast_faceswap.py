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

def main():
    t0 = time.time()
    print("Loading track and source image...")
    track = np.load(TRACK_PATH)
    cx_arr = track['cx']
    cy_arr = track['cy']

    sai_img = cv2.imread(SAI_IMG_PATH)

    print("Extracting Sai face embeddings...")
    app_full = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app_full.prepare(ctx_id=0, det_size=(640, 640))
    sai_faces = app_full.get(sai_img)
    if not sai_faces:
        raise RuntimeError("No face detected in Sai image!")
    sai_source = sai_faces[0]
    print(f"Extracted Sai face (norm={np.linalg.norm(sai_source.embedding):.2f})")

    print("Setting up fast detector and inswapper...")
    app_fast = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    app_fast.prepare(ctx_id=0, det_size=(256, 256))

    swapper_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(SRC_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    print(f"Processing {total_frames} frames ({w}x{h} at {fps} fps)...")
    last_valid_face = None
    pad = 140

    frame_start = time.time()
    for i in range(total_frames):
        ok, fr = cap.read()
        if not ok:
            break

        cx = int(cx_arr[i])
        cy = int(cy_arr[i])

        x0 = max(0, cx - pad)
        y0 = max(0, cy - pad)
        x1 = min(w, cx + pad)
        y1 = min(h, cy + pad)

        roi = fr[y0:y1, x0:x1]
        faces = app_fast.get(roi)

        if faces:
            # Pick largest
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            # Shift back
            face.bbox[0] += x0
            face.bbox[2] += x0
            face.bbox[1] += y0
            face.bbox[3] += y0
            face.kps[:, 0] += x0
            face.kps[:, 1] += y0
            last_valid_face = face
            swapped = swapper.get(fr, face, sai_source, paste_back=True)
        elif last_valid_face is not None:
            swapped = swapper.get(fr, last_valid_face, sai_source, paste_back=True)
        else:
            swapped = fr

        out.write(swapped)

        if (i + 1) % 20 == 0 or i == total_frames - 1:
            elapsed = time.time() - frame_start
            fps_proc = (i + 1) / elapsed
            rem = (total_frames - (i + 1)) / max(fps_proc, 1e-3)
            print(f"[{i+1:3d}/{total_frames}] {fps_proc:.1f} fps (est {rem:.0f}s remaining)...", flush=True)

    cap.release()
    out.release()
    total_time = time.time() - t0
    print(f"SUCCESS! Created {OUT_VIDEO} in {total_time:.1f}s")

if __name__ == "__main__":
    main()
