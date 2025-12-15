import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent_a import AgentA


def debug_small_image(image_path):
    print(f"DEBUGGING IMAGE: {image_path}")

    # 1. Load & Resize
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    target_h = 48
    scale = target_h / h
    new_w = int(w * scale)
    img_resized = cv2.resize(img, (new_w, target_h))
    print(f"Resized: {w}x{h} -> {new_w}x{target_h}")

    # 2. Slice Windows (Simple)
    win_size = 128
    stride = 64  # More overlap

    agent = AgentA(use_gpu=True)

    print("\n--- RAW WINDOW OUTPUTS ---")
    for x in range(0, new_w, stride):
        end = min(x + win_size, new_w)
        window = img_resized[:, x:end]

        # 强制纯识别
        rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
        res = agent.inference(rgb, skip_detection=True)

        text = res[0]["text"] if res else "[EMPTY]"
        print(f"Window [{x}:{end}] -> '{text}'")

        if end >= new_w:
            break


if __name__ == "__main__":
    # 请替换为你的图片路径
    debug_small_image("my_test_image.jpg")
