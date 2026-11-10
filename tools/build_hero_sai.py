import os
import math
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = r"C:\Users\hmind\.gemini\antigravity\scratch\cinematic-portfolio"
SRC_CUTOUT = os.path.join(ROOT, "sai_cutout.png")
MEDIA = os.path.join(ROOT, "public", "media")
OUT_MP4 = os.path.join(MEDIA, "hero.mp4")
OUT_WEBM = os.path.join(MEDIA, "hero.webm")
OUT_JPG = os.path.join(MEDIA, "hero.jpg")

W_HALF = 720
H = 1280
TOTAL_W = W_HALF * 2  # 1440
FRAMES = 240
FPS = 24

def build():
    print("Loading base cutout...")
    base = Image.open(SRC_CUTOUT)
    bbox = base.getbbox()
    subject = base.crop(bbox)
    sw, sh = subject.size

    # Fit subject to approximately 82% of height
    target_h = int(H * 0.82)
    scale = target_h / float(sh)
    target_w = int(sw * scale)
    base_resized = subject.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # Base array
    subj_arr = np.array(base_resized, dtype=np.float32)

    # Temporary directory for video pipe
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{TOTAL_W}x{H}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-movflags", "+faststart",
        "-crf", "18",
        "-preset", "medium",
        OUT_MP4
    ]
    
    print("Starting ffmpeg process for hero.mp4...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    poster_saved = False

    for i in range(FRAMES):
        t = float(i) / float(FRAMES)
        phase = 2.0 * math.pi * t

        # Gentle breathing motion: continuous and perfectly seamless loop
        cur_scale = 1.0 + 0.018 * (1.0 - math.cos(phase)) / 2.0
        cur_dy = math.sin(phase) * 8.0
        cur_dx = math.sin(phase * 0.5) * 3.0

        fw = int(target_w * cur_scale)
        fh = int(target_h * cur_scale)
        
        frame_subj = cv2.resize(subj_arr, (fw, fh), interpolation=cv2.INTER_LINEAR)
        
        # Color buffer (720x1280) and Matte buffer (720x1280)
        col_buf = np.zeros((H, W_HALF, 3), dtype=np.float32)
        mat_buf = np.zeros((H, W_HALF), dtype=np.float32)

        # Anchor feet near 0.94 of H
        feet_y = int(H * 0.94 + cur_dy)
        top_y = feet_y - fh
        left_x = int((W_HALF - fw) / 2.0 + cur_dx)
        
        # Clip coordinates
        src_y0 = max(0, -top_y)
        src_x0 = max(0, -left_x)
        dst_y0 = max(0, top_y)
        dst_x0 = max(0, left_x)
        
        copy_h = min(fh - src_y0, H - dst_y0)
        copy_w = min(fw - src_x0, W_HALF - dst_x0)

        if copy_h > 0 and copy_w > 0:
            slice_rgba = frame_subj[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
            alpha = slice_rgba[:, :, 3] / 255.0
            
            # RGB color grade
            rgb = slice_rgba[:, :, :3]
            # Boost contrast and richness
            rgb = np.clip((rgb - 128.0) * 1.1 + 128.0, 0, 255)
            # Premultiply by alpha for the compositor
            for c in range(3):
                col_buf[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w, c] = rgb[:, :, c] * alpha
                
            mat_buf[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = slice_rgba[:, :, 3]

        # Combine into packed side-by-side frame: left color, right matte
        packed = np.zeros((H, TOTAL_W, 3), dtype=np.uint8)
        # Convert RGB to BGR for ffmpeg
        packed[:, :W_HALF, 0] = np.clip(col_buf[:, :, 2], 0, 255).astype(np.uint8) # B
        packed[:, :W_HALF, 1] = np.clip(col_buf[:, :, 1], 0, 255).astype(np.uint8) # G
        packed[:, :W_HALF, 2] = np.clip(col_buf[:, :, 0], 0, 255).astype(np.uint8) # R

        # Matte on right half in all 3 channels
        m_u8 = np.clip(mat_buf, 0, 255).astype(np.uint8)
        packed[:, W_HALF:, 0] = m_u8
        packed[:, W_HALF:, 1] = m_u8
        packed[:, W_HALF:, 2] = m_u8

        # Save poster
        if not poster_saved:
            cv2.imwrite(OUT_JPG, packed[:, :W_HALF, :], [cv2.IMWRITE_JPEG_QUALITY, 88])
            poster_saved = True

        proc.stdin.write(packed.tobytes())

    proc.stdin.close()
    proc.wait()
    print("hero.mp4 finished!")

    # Encode hero.webm as well
    print("Encoding hero.webm...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", OUT_MP4,
        "-c:v", "libvpx-vp9",
        "-b:v", "1800k",
        "-crf", "28",
        OUT_WEBM
    ], check=True)
    print("hero.webm finished!")

if __name__ == "__main__":
    build()
