import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

ROOT = r"C:\Users\hmind\.gemini\antigravity\scratch\cinematic-portfolio"
SRC_VIDEO = os.path.join(ROOT, "assets_src", "v1.mp4")
OUT_VIDEO = os.path.join(ROOT, "assets_src", "v1_sai.mp4")
SAI_IMG_PATH = r"C:\Users\hmind\Downloads\sai image.jpeg"

def main():
    print("Initializing FaceAnalysis...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
    print("Loading inswapper model...")
    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=['CPUExecutionProvider'])

    print("Extracting source face (Sai)...")
    sai_img = cv2.imread(SAI_IMG_PATH)
    sai_faces = app.get(sai_img)
    if len(sai_faces) == 0:
        raise RuntimeError("No face detected in Sai's image!")
    sai_source = sai_faces[0]
    print(f"Source face detected! Bbox: {sai_source.bbox}")

    cap = cv2.VideoCapture(SRC_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    print(f"Processing {total_frames} frames ({w}x{h} at {fps} fps)...")

    last_target_face = None

    for i in range(total_frames):
        ok, fr = cap.read()
        if not ok:
            break

        # Detect face in current frame
        target_faces = app.get(fr)
        if len(target_faces) > 0:
            # Pick the face closest to previous or largest
            best_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            last_target_face = best_face
            res = swapper.get(fr, best_face, sai_source, paste_back=True)
        elif last_target_face is not None:
            res = swapper.get(fr, last_target_face, sai_source, paste_back=True)
        else:
            res = fr

        out.write(res)
        if i % 30 == 0 or i == total_frames - 1:
            print(f"Processed frame {i}/{total_frames}...", flush=True)

    cap.release()
    out.release()
    print("Finished generating AI Face-Swapped Walking Video ->", OUT_VIDEO)

if __name__ == "__main__":
    main()
