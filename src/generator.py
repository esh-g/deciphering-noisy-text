from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

import noise
ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "configs" / "generator.yaml"

def load_configs(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg 



def sample_text(cfg, rng: np.random.Generator):
    charset = cfg["charset"]
    text_len = rng.integers(cfg["min_len"], cfg["max_len"] + 1)
    idx = rng.integers(0, len(charset), size=text_len)
    return "".join(charset[i] for i in idx)


def pick_font(cfg, rng, split="train"):
    if split == "train":
        names = cfg["train_fonts"]
    elif split == "heldout" or split == 'test':
        names = cfg["heldout_fonts"]

    name = names[rng.integers(0, len(names))]
    size = rng.integers(cfg["font_size_range"][0], cfg["font_size_range"][1] + 1)
    return name, size


@lru_cache(maxsize=256)
def get_font(path, size):
    
    return ImageFont.truetype(str(path), size)


def _render_glyph(ch, font, font_size, height, y_offset, angle):
    
    canvas = Image.new("L", (font_size * 2, height), 0)
    draw = ImageDraw.Draw(canvas)

    draw.text((font_size // 2, height // 2 + y_offset), ch,
              fill=255, font=font, anchor="lm")
    canvas = canvas.rotate(angle, resample=Image.BICUBIC)

    bbox = canvas.getbbox()                  
    if bbox is None:                         
        return np.zeros((height, max(font_size // 3, 1)), dtype=np.float32)
    canvas = canvas.crop((bbox[0], 0, bbox[2], height))   
    return np.asarray(canvas, dtype=np.float32) / 255.0

def render_text(text, font_name, font_size, cfg, rng):
    H = cfg["image_height"]
    margin = cfg["margin"]
    jit = cfg["char_jitter"]
    font = get_font(ROOT / cfg["font_dir"] / font_name, font_size)

    bg = int(rng.integers(cfg["background"]["gray_min"], cfg["background"]["gray_max"] + 1))
    fg = int(rng.integers(cfg["text_gray"]["min"], cfg["text_gray"]["max"] + 1))

    glyphs = []
    for ch in text:
        y_off = int(rng.integers(-jit["y_pixels"], jit["y_pixels"] + 1))
        angle = float(rng.uniform(-jit["rotate_degrees"], jit["rotate_degrees"]))
        glyphs.append(_render_glyph(ch, font, font_size, H, y_off, angle))

    lo, hi = jit["spacing_pixels"]
    gaps = rng.integers(lo, hi + 1, size=len(glyphs) - 1)

    widths = [g.shape[1] for g in glyphs]
    W = 2 * margin + sum(widths) + int(gaps.sum())

    mask = np.zeros((H, W), dtype=np.float32)
    x = margin
    for i, g in enumerate(glyphs):
        w = g.shape[1]
        mask[:, x:x + w] = np.maximum(mask[:, x:x + w], g)   
        x += w + (int(gaps[i]) if i < len(gaps) else 0)

    img = bg * (1.0 - mask) + fg * mask
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_noise(img, steps, rng):
    
    for step in steps:
        params = dict(step)                 
        func = getattr(noise, params.pop("fn"))
        img = func(img, rng=rng, **params)
    return img


def generate_sample(cfg, level, rng, split="train"):
    text = sample_text(cfg, rng)
    font_name, font_size = pick_font(cfg, rng, split)
    clean = render_text(text, font_name, font_size, cfg, rng)
    noisy = apply_noise(clean, cfg["levels"][level], rng)

    meta = {
        "text": text,
        "level": level,
        "font": font_name,
        "font_size": font_size,
        "split": split,
        "height": noisy.shape[0],
        "width": noisy.shape[1],
    }
    return noisy, text, meta


def generate_batch(cfg, n, level, seed, split="train", start_index=0):
    samples = []
    for i in range(start_index, start_index + n):
        rng = np.random.default_rng([seed, i])   
        img, text, meta = generate_sample(cfg, level, rng, split)
        meta["seed"] = seed
        meta["index"] = i
        samples.append({"image": img, "text": text, "meta": meta})
    return samples


if __name__ == "__main__":
    cfg = load_configs(CONFIG_PATH)
    
    for level in cfg["levels"]:
        s = generate_batch(cfg, 1, level, seed=0)[0]
        Image.fromarray(s["image"]).save(ROOT / "data" / f"sample_level{level}.png")
        print(level, s["text"], s["image"].shape)