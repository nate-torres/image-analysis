import cv2
import numpy as np
from collections import Counter

# ---------- CONSTANTS ----------
# Hue intervals in OpenCV scale (0–179) for each base color
BASE_COLOR_RANGES = {
    "red":      [(0, 10), (170, 179)],
    "orange":   [(11, 25)],
    "yellow":   [(26, 35)],
    "green":    [(36, 85)],
    "cyan":     [(86, 100)],
    "blue":     [(101, 130)],
    "purple":   [(131, 150)],
    "magenta":  [(151, 169)],
}
SAT_THRESH = 25
BLACK_V, WHITE_V = 50, 205
MIN_PCT = 0.02          # 2% of the pixels required

# ---------- HUE → BASE‐COLOR MAPPING ----------
def hue_to_base(h):
    """Map a single HSV hue value to one of our base colors."""
    for name, intervals in BASE_COLOR_RANGES.items():
        for lo, hi in intervals:
            # straightforward range
            if lo <= hi and lo <= h <= hi:
                return name
            # wrap-around range (e.g. red 170→179 + 0→10)
            if lo > hi and (h >= lo or h <= hi):
                return name
    return None  # should not happen

# ---------- PALETTE EXTRACTION ----------
def extract_base_palette(path, samp=160, min_pct=MIN_PCT):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    # downsample for speed
    bgr = cv2.resize(bgr, (samp, samp))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    counts = Counter()
    # count black/white/gray vs. color by hue
    for h, s, v in hsv.reshape(-1, 3):
        if s < SAT_THRESH:
            if   v < BLACK_V:  counts["black"] += 1
            elif v > WHITE_V:  counts["white"] += 1
            
        else:
            base = hue_to_base(int(h))
            counts[base] += 1

    total = samp * samp
    # keep only colors >= the absolute threshold
    palette = sorted([c for c, cnt in counts.items() if cnt / total >= min_pct])
    return palette

# Example subset of harmony definitions (use full list in production)
HARMONIES = [
    { "colors": ["red"], "harmony": "monochromatic" },
  { "colors": ["orange"], "harmony": "monochromatic" },
  { "colors": ["yellow"], "harmony": "monochromatic" },
  { "colors": ["green"], "harmony": "monochromatic" },
  { "colors": ["cyan"], "harmony": "monochromatic" },
  { "colors": ["blue"], "harmony": "monochromatic" },
  { "colors": ["purple"], "harmony": "monochromatic" },
  { "colors": ["magenta"], "harmony": "monochromatic" },
  { "colors": ["black"], "harmony": "monochromatic" },
  { "colors": ["white"], "harmony": "monochromatic" },
  { "colors": ["gray"], "harmony": "monochromatic" },
  { "colors": ["black", "white"], "harmony": "monochromatic" },
  { "colors": ["black", "gray"], "harmony": "monochromatic" },
  { "colors": ["white", "gray"], "harmony": "monochromatic" },
  { "colors": ["white", "gray", "black"], "harmony": "monochromatic" },


  { "colors": ["cyan", "red"], "harmony": "complementary" },
  { "colors": ["cyan", "orange"], "harmony": "complementary" },
  { "colors": ["blue", "red"], "harmony": "complementary" },
  { "colors": ["blue", "orange"], "harmony": "complementary" },
  { "colors": ["purple", "yellow"], "harmony": "complementary" },
  { "colors": ["green", "magenta"], "harmony": "complementary" },
  { "colors": ["green", "purple"], "harmony": "complementary" },
  { "colors": ["green", "red"], "harmony": "complementary" },


  { "colors": ["orange", "red"], "harmony": "analogous" },
  { "colors": ["orange", "yellow"], "harmony": "analogous" },
  { "colors": ["green", "orange"], "harmony": "analogous" },
  { "colors": ["green", "yellow"], "harmony": "analogous" },
  { "colors": ["cyan", "green"], "harmony": "analogous" },
  { "colors": ["cyan", "magenta"], "harmony": "analogous" },
  { "colors": ["cyan", "purple"], "harmony": "analogous" },
  { "colors": ["cyan", "yellow"], "harmony": "analogous" },
  { "colors": ["blue", "cyan"], "harmony": "analogous" },
  { "colors": ["blue", "green"], "harmony": "analogous" },
  { "colors": ["blue", "magenta"], "harmony": "analogous" },
  { "colors": ["blue", "purple"], "harmony": "analogous" },
  { "colors": ["magenta", "orange"], "harmony": "analogous" },
  { "colors": ["magenta", "purple"], "harmony": "analogous" },
  { "colors": ["magenta", "red"], "harmony": "analogous" },
  { "colors": ["red", "yellow"], "harmony": "analogous" },
  { "colors": ["red", "purple"], "harmony": "analogous" },


  { "colors": ["orange", "red", "yellow"], "harmony": "analogous" },
  { "colors": ["magenta", "orange", "red"], "harmony": "analogous" },
  { "colors": ["magenta", "purple", "red"], "harmony": "analogous" },
  { "colors": ["blue", "magenta", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "green"], "harmony": "analogous" },
  { "colors": ["cyan", "green", "yellow"], "harmony": "analogous" },
  { "colors": ["green", "orange", "yellow"], "harmony": "analogous" },

  { "colors": ["magenta", "orange", "yellow"], "harmony": "analogous" },
  { "colors": ["orange", "purple", "red"], "harmony": "analogous" },
  { "colors": ["blue", "magenta", "red"], "harmony": "analogous" },
  { "colors": ["cyan", "magenta", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "green", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "yellow"], "harmony": "analogous" },
  { "colors": ["cyan", "green", "orange"], "harmony": "analogous" },
  { "colors": ["green", "red", "yellow"], "harmony": "analogous" },

  { "colors": ["green", "red", "orange"], "harmony": "analogous" },
  { "colors": ["magenta", "red", "yellow"], "harmony": "analogous" },
  { "colors": ["magenta", "orange", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "purple", "red"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "magenta"], "harmony": "analogous" },
  { "colors": ["cyan", "green", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "green", "yellow"], "harmony": "analogous" },
  { "colors": ["cyan", "orange", "yellow"], "harmony": "analogous" },


  { "colors": ["green", "orange", "red", "yellow"], "harmony": "analogous" },
  { "colors": ["magenta", "orange", "red", "yellow"], "harmony": "analogous" },
  { "colors": ["magenta", "orange", "purple", "red"], "harmony": "analogous" },
  { "colors": ["blue", "magenta", "purple", "red"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "magenta", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "green", "purple"], "harmony": "analogous" },
  { "colors": ["blue", "cyan", "green", "yellow"], "harmony": "analogous" },
  { "colors": ["cyan", "green", "orange", "yellow"], "harmony": "analogous" },


  { "colors": ["green", "orange", "purple"], "harmony": "triadic" },
  { "colors": ["cyan", "magenta", "orange"], "harmony": "triadic" },
  { "colors": ["blue", "red", "yellow"], "harmony": "triadic" },
  { "colors": ["blue", "green", "red"], "harmony": "triadic" },
  { "colors": ["green", "purple", "red"], "harmony": "triadic" },
  { "colors": ["green", "magenta", "orange"], "harmony": "triadic" },
  { "colors": ["cyan", "red", "yellow"], "harmony": "triadic" },
  { "colors": ["cyan", "magenta", "orange"], "harmony": "triadic" },


  { "colors": ["blue", "orange", "red"], "harmony": "split complementary" },
  { "colors": ["blue", "orange", "yellow"], "harmony": "split complementary" },
  { "colors": ["cyan", "orange", "red"], "harmony": "split complementary" },
  { "colors": ["green", "magenta", "red"], "harmony": "split complementary" },
  { "colors": ["green", "magenta", "purple"], "harmony": "split complementary" },
  { "colors": ["green", "orange", "red"], "harmony": "split complementary" },
  { "colors": ["blue", "purple", "yellow"], "harmony": "split complementary" },
  { "colors": ["blue", "orange", "purple"], "harmony": "split complementary" },
  { "colors": ["blue", "cyan", "orange"], "harmony": "split complementary" },
  { "colors": ["cyan", "green", "red"], "harmony": "split complementary" },
  { "colors": ["green", "magenta", "yellow"], "harmony": "split complementary" },
  { "colors": ["green", "purple", "yellow"], "harmony": "split complementary" },
  { "colors": ["blue", "cyan", "red"], "harmony": "split complementary" },
  { "colors": ["blue", "green", "orange"], "harmony": "split complementary" },
  { "colors": ["cyan", "green", "magenta"], "harmony": "split complementary" },
  { "colors": ["cyan", "magenta", "red"], "harmony": "split complementary" },
  { "colors": ["magenta", "purple", "yellow"], "harmony": "split complementary" },
  { "colors": ["orange", "purple", "yellow"], "harmony": "split complementary" },
  { "colors": ["blue", "cyan", "green", "red"], "harmony": "split complementary" },


  {
    "colors": ["blue", "cyan", "orange", "purple"],
    "harmony": "split complementary",
  },
  {
    "colors": ["blue", "magenta", "purple", "yellow"],
    "harmony": "split complementary",
  },
  {
    "colors": ["green", "magenta", "purple", "red"],
    "harmony": "split complementary",
  },
  {
    "colors": ["cyan", "magenta", "orange", "red"],
    "harmony": "split complementary",
  },
  {
    "colors": ["cyan", "orange", "red", "yellow"],
    "harmony": "split complementary",
  },
  {
    "colors": ["blue", "orange", "red", "yellow"],
    "harmony": "split complementary",
  },
  {
    "colors": ["cyan", "green", "magenta", "yellow"],
    "harmony": "split complementary",
  },


  {
    "colors": ["blue", "magenta", "orange", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["blue", "orange", "purple", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["blue", "orange", "purple", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["blue", "green", "purple", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["green", "orange", "purple", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["green", "magenta", "orange", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["green", "magenta", "red", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["green", "purple", "red", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["cyan", "green", "magenta", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["blue", "green", "orange", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["cyan", "green", "orange", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["cyan", "orange", "purple", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["magenta", "orange", "purple", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["blue", "cyan", "orange", "red"],
    "harmony": "double complementary",
  },
  {
    "colors": ["blue", "cyan", "orange", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["cyan", "purple", "red", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["green", "magenta", "purple", "yellow"],
    "harmony": "double complementary",
  },
  {
    "colors": ["cyan", "magenta", "orange", "yellow"],
    "harmony": "double complementary",
  },


  { "colors": ["cyan", "green", "orange", "purple"], "harmony": "square" },
  { "colors": ["cyan", "green", "magenta", "purple"], "harmony": "square" },
  { "colors": ["cyan", "green", "purple", "red"], "harmony": "square" },
  { "colors": ["cyan", "purple", "red", "yellow"], "harmony": "square" },
  { "colors": ["green", "orange", "purple", "yellow"], "harmony": "square" },
  { "colors": ["blue", "green", "orange", "yellow"], "harmony": "square" },
  { "colors": ["blue", "green", "magenta", "orange"], "harmony": "square" },
  { "colors": ["cyan", "green", "magenta", "orange"], "harmony": "square" },
  { "colors": ["blue", "green", "orange", "purple"], "harmony": "square" },
  { "colors": ["blue", "green", "red", "yellow"], "harmony": "square" },
]

def find_color_harmonies(detected):
    d = set(detected)
    return [h for h in HARMONIES if set(h["colors"]).issubset(d)]

def best_harmony(matches):
    return max(matches, key=lambda h: len(h["colors"])) if matches else None

