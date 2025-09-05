# pixel_mushroom.py
# Generate Mario-like pixel mushrooms (transparent PNG, centered).
# - Cap is a true HALF-CIRCLE (semi-ellipse) + flat base rows (no interior gap).
# - Solid black outer outline.
# - Full control from code or CLI (notebook-safe).

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import argparse, os
# --- at the top (imports) ---

# ======================= Palettes =======================
class Palette:
    def __init__(self, cap, cap_shadow, spot, stem, stem_shadow, ring, outline, eye):
        self.cap = cap; self.cap_shadow = cap_shadow; self.spot = spot
        self.stem = stem; self.stem_shadow = stem_shadow; self.ring = ring
        self.outline = outline; self.eye = eye

PALETTE_RED = Palette((220,44,44,255),(180,24,24,255),(255,255,255,255),
                      (240,210,170,255),(210,180,142,255),(250,230,190,255),
                      (0,0,0,255),(0,0,0,255))
PALETTE_GREEN = Palette((52,180,80,255),(34,140,58,255),(255,255,255,255),
                        (240,210,170,255),(210,180,142,255),(250,230,190,255),
                        (0,0,0,255),(0,0,0,255))
PALETTE_BLUE = Palette((40,120,220,255),(28,90,170,255),(240,240,255,255),
                       (240,210,170,255),(210,180,142,255),(250,230,190,255),
                       (0,0,0,255),(0,0,0,255))
PALETTES = {"red": PALETTE_RED, "green": PALETTE_GREEN, "blue": PALETTE_BLUE}

# ======================= Color helpers =======================
def parse_color(value):
    """
    Accepts '#RRGGBB', '#RRGGBBAA', '#RGB', '#RGBA', or 'r,g,b[,a]' and returns (r,g,b,a).
    Pass None to keep defaults.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        t = tuple(int(x) for x in value)
        return (t[0], t[1], t[2], t[3] if len(t) > 3 else 255)
    s = str(value).strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:  # RGB
            r, g, b = [int(c*2, 16) for c in h]
            return (r, g, b, 255)
        if len(h) == 4:  # RGBA
            r, g, b, a = [int(c*2, 16) for c in h]
            return (r, g, b, a)
        if len(h) == 6:  # RRGGBB
            r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
            return (r, g, b, 255)
        if len(h) == 8:  # RRGGBBAA
            r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16); a = int(h[6:8], 16)
            return (r, g, b, a)
        raise ValueError(f"Bad hex color: {s}")
    parts = [int(x) for x in s.replace(" ", "").split(",")]
    if len(parts) == 3:
        parts.append(255)
    if len(parts) != 4:
        raise ValueError(f"Bad rgba color: {s}")
    return tuple(parts[:4])



# ======================= Pixel helpers =======================
def new_canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)

def put_pixel(img: np.ndarray, x: int, y: int, color: Tuple[int,int,int,int]):
    h, w, _ = img.shape
    if 0 <= x < w and 0 <= y < h: img[y, x] = color

def draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int,int,int,int]):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            put_pixel(img, x, y, color)

def draw_semiellipse_mask(width: int, height: int, flat_on_bottom: bool = True) -> np.ndarray:
    """Filled SEMI-ELLIPSE; if flat_on_bottom, keep top half."""
    a = width / 2.0
    b = height / 2.0
    cx, cy = a - 0.5, b - 0.5
    yy, xx = np.mgrid[0:height, 0:width]
    ellipse = ((xx - cx)**2)/(a*a) + ((yy - cy)**2)/(b*b) <= 1.0
    return ellipse & (yy <= cy) if flat_on_bottom else ellipse & (yy >= cy)

def outline_mask(mask: np.ndarray) -> np.ndarray:
    """1px outline around True regions (4-neighborhood)."""
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not mask[y, x]: continue
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x+dx, y+dy
                if not(0 <= nx < w and 0 <= ny < h) or not mask[ny, nx]:
                    out[y, x] = True; break
    return out

# ======================= Parameters =======================
@dataclass
class MushroomParams:
    # --- Color overrides (optional). If set, override palette_name defaults ---
    cap_color:       Optional[Tuple[int,int,int,int]] = None
    cap_shadow_color:Optional[Tuple[int,int,int,int]] = None
    spot_color:      Optional[Tuple[int,int,int,int]] = None
    stem_color:      Optional[Tuple[int,int,int,int]] = None
    stem_shadow_color:Optional[Tuple[int,int,int,int]] = None
    ring_color:      Optional[Tuple[int,int,int,int]] = None
    outline_color:   Optional[Tuple[int,int,int,int]] = None
    eye_color:       Optional[Tuple[int,int,int,int]] = None

    # Canvas
    grid_w: int = 24
    grid_h: int = 24
    centered: bool = True
    offset_x: int = 0
    offset_y: int = 0

    # Cap (semi-ellipse + flat base)
    cap_w: int = 22          # wider default
    cap_h: int = 14          # more arc rows -> thicker semi-circle
    cap_flat: int = 2        # flat rows under the arc
    cap_round: float = 1.25  # NEW: >1 = rounder/taller arc, <1 = flatter arc

    # Stem (forced odd width for perfect symmetry)
    stem_w: int = 9
    stem_h: int = 8

    # Ring
    ring_thickness: int = 0  # 0 to hide

    # Spots
    spot_radius: int = 2
    add_center_spot: bool = True
    add_side_spots: bool = True

    # Face
    add_eyes: bool = False
    eye_height_from_top_of_stem: int = 2
    eye_gap: int = 2

    # Style
    palette_name: str = "green"  # "red"|"green"|"blue"
    add_outlines: bool = True
    shadow_on_bottom_of_cap: bool = True

    # Output
    scale: int = 16

    def palette(self) -> "Palette":
        base = PALETTES.get(self.palette_name, PALETTE_RED)
        def pick(override, default): return override if override is not None else default
        return Palette(
            pick(self.cap_color,        base.cap),
            pick(self.cap_shadow_color, base.cap_shadow),
            pick(self.spot_color,       base.spot),
            pick(self.stem_color,       base.stem),
            pick(self.stem_shadow_color,base.stem_shadow),
            pick(self.ring_color,       base.ring),
            pick(self.outline_color,    base.outline),
            pick(self.eye_color,        base.eye),
        )


# ======================= Geometry / Drawing =======================
def auto_tune_mario_like(p: MushroomParams) -> MushroomParams:
    """Gentle nudges for proportions + canvas safety.
    - Do NOT mutate cap_w from cap_h (no hidden coupling).
    - Stem is independent, only constrained to not exceed cap width.
    """
    p = MushroomParams(**asdict(p))  # copy

    # Effective arc height (cap roundness)
    eff_cap_h = max(3, int(round(p.cap_h * p.cap_round)))

    # DO NOT touch p.cap_w here. (Removing the aspect-based widening prevents cap_h from influencing cap_w.)

    # ---- Stem: independent of cap, but never wider than the cap ----
    p.stem_w = max(2, int(round(p.stem_w)))   # allow even numbers, min 2
    p.stem_h = max(1, int(round(p.stem_h)))

    # clamp to cap width (equal allowed)
    if p.stem_w > p.cap_w:
        p.stem_w = p.cap_w

    # force even width (for clean centering) — if odd, bump up by 1 but keep ≤ cap_w
    if p.stem_w % 2 == 1:
        if p.stem_w < p.cap_w:
            p.stem_w += 1
        else:
            p.stem_w -= 1  # if already at cap limit and odd, step down


    # ---- Canvas safety margin ----
    est_total_h = eff_cap_h + p.cap_flat + p.stem_h + 4
    widest = max(p.cap_w, p.stem_w)
    p.grid_w = max(p.grid_w, widest + 4)
    p.grid_h = max(p.grid_h, est_total_h)

    # Store the effective arc height (used by draw)
    p._eff_cap_h = eff_cap_h  # type: ignore[attr-defined]
    return p



def draw_mushroom(p: MushroomParams) -> Image.Image:
    p = auto_tune_mario_like(p)
    img = new_canvas(p.grid_w, p.grid_h)
    pal = p.palette()

    cap_w = p.cap_w
    cap_h_eff = int(getattr(p, "_eff_cap_h"))  # applied roundness

    # Build cap arc (semi-ellipse) with effective height
    cap_mask_local = draw_semiellipse_mask(cap_w, cap_h_eff, flat_on_bottom=True)
    rows_with_pixels = np.where(cap_mask_local.any(axis=1))[0]
    last_row = int(rows_with_pixels.max()) if len(rows_with_pixels) else -1
    cap_draw_h = (last_row + 1) + p.cap_flat
    cap_draw_w = cap_w

    # Stem
    stem_draw_h = p.stem_h
    stem_draw_w = p.stem_w

    # Bounding box (cap + stem)
    bbox_w = max(cap_draw_w, stem_draw_w)
    bbox_h = cap_draw_h + stem_draw_h

    # Placement (center + offsets)
    if p.centered:
        bbox_left = (p.grid_w - bbox_w) // 2
        bbox_top  = (p.grid_h - bbox_h) // 2
    else:
        bbox_left = 0; bbox_top = 0
    bbox_left += p.offset_x; bbox_top += p.offset_y

    cap_left = bbox_left + (bbox_w - cap_draw_w) // 2
    cap_top  = bbox_top
    # Center the stem under the cap precisely (handles even/odd widths correctly)
    # New: center stem under the cap’s true middle (handles even/odd widths)
    cap_mid = cap_left + (cap_draw_w - 1) / 2.0
    stem_left = int(round(cap_mid - (stem_draw_w - 1) / 2.0))


    stem_top  = cap_top + cap_draw_h

    # ---- CAP: arc + flat base directly under the actual ellipse
    for y in range(cap_h_eff):
        for x in range(cap_w):
            if cap_mask_local[y, x]:
                put_pixel(img, cap_left + x, cap_top + y, pal.cap)

    base_y0 = cap_top + last_row + 1
    for y in range(base_y0, base_y0 + p.cap_flat):
        for x in range(cap_left, cap_left + cap_w):
            put_pixel(img, x, y, pal.cap)

    if p.shadow_on_bottom_of_cap and p.cap_flat > 0:
        y = base_y0 + p.cap_flat - 1
        for x in range(cap_left + 1, cap_left + cap_w - 1):
            if img[y, x, 3] != 0:
                put_pixel(img, x, y, pal.cap_shadow)

    # ---- STEM under cap base (symmetry: odd width, perfectly centered)
    stem_right = stem_left + p.stem_w - 1
    stem_bottom = stem_top + p.stem_h - 1
    draw_rect(img, stem_left, stem_top, stem_right, stem_bottom, pal.stem)

    shade_x = stem_right if p.stem_w >= 5 else stem_left
    for y in range(stem_top, stem_bottom + 1):
        put_pixel(img, shade_x, y, pal.stem_shadow)

    # ---- Optional ring
    if p.ring_thickness > 0:
        ry0 = stem_top + 1
        for t in range(p.ring_thickness):
            y = ry0 + t
            for x in range(stem_left - 1, stem_right + 2):
                put_pixel(img, x, y, pal.ring)

    # ---- Spots on cap pixels
    def draw_spot(cx_pix: int, cy_pix: int, r: int):
        for y in range(cy_pix - r, cy_pix + r + 1):
            for x in range(cx_pix - r, cx_pix + r + 1):
                if (x - cx_pix)**2 + (y - cy_pix)**2 <= r*r:
                    if 0 <= x < p.grid_w and 0 <= y < p.grid_h and img[y, x, 3] != 0:
                        put_pixel(img, x, y, pal.spot)

    if p.add_center_spot:
        draw_spot(cap_left + cap_w // 2, cap_top + (last_row // 2) + 1, p.spot_radius)
    if p.add_side_spots:
        offset_x = max(3, cap_w // 4)
        draw_spot(cap_left + cap_w // 2 - offset_x, cap_top + (last_row // 2), max(1, p.spot_radius - 1))
        draw_spot(cap_left + cap_w // 2 + offset_x, cap_top + (last_row // 2), max(1, p.spot_radius - 1))

    # ---- Eyes (optional)
    if p.add_eyes:
        cx_mid = bbox_left + bbox_w // 2
        ey = stem_top + p.eye_height_from_top_of_stem
        ex1 = cx_mid - (p.eye_gap // 2) - 1
        ex2 = cx_mid + (p.eye_gap // 2) + 1
        for dy in (0, 1):
            if stem_top <= ey + dy <= stem_bottom:
                put_pixel(img, ex1, ey + dy, pal.outline)
                put_pixel(img, ex2, ey + dy, pal.outline)

    # ---- OUTER OUTLINE (solid black)
    if p.add_outlines:
        cap_mask_full = np.zeros((p.grid_h, p.grid_w), dtype=bool)
        cap_mask_full[cap_top:cap_top + cap_h_eff, cap_left:cap_left + cap_w] = cap_mask_local
        cap_mask_full[base_y0:base_y0 + p.cap_flat, cap_left:cap_left + cap_w] = True

        stem_mask_full = np.zeros((p.grid_h, p.grid_w), dtype=bool)
        stem_mask_full[stem_top:stem_bottom + 1, stem_left:stem_right + 1] = True

        shroom_mask = cap_mask_full | stem_mask_full
        outline = outline_mask(shroom_mask)
        ys, xs = np.where(outline)
        for y, x in zip(ys, xs):
            put_pixel(img, x, y, pal.outline)

    return Image.fromarray(img, mode="RGBA")

def save_upscaled(img: Image.Image, scale: int, path: str):
    up = img.resize((img.width * scale, img.height * scale), resample=Image.Resampling.NEAREST)
    up.save(path)

# ======================= CLI (Notebook-safe) =======================
def build_from_args(args) -> MushroomParams:
    cfg = MushroomParams()
    # existing numeric/bool fields...
    for f in [
        "grid_w","grid_h","centered","offset_x","offset_y",
        "cap_w","cap_h","cap_flat","cap_round",
        "stem_w","stem_h","ring_thickness","spot_radius",
        "add_center_spot","add_side_spots","add_eyes",
        "eye_height_from_top_of_stem","eye_gap","palette_name",
        "add_outlines","shadow_on_bottom_of_cap","scale"
    ]:
        if getattr(args, f, None) is not None:
            setattr(cfg, f, getattr(args, f))

    # color fields (strings -> RGBA tuples)
    color_fields = [
        "cap_color","cap_shadow_color","spot_color","stem_color",
        "stem_shadow_color","ring_color","outline_color","eye_color"
    ]
    for f in color_fields:
        val = getattr(args, f, None)
        if val is not None:
            setattr(cfg, f, parse_color(val))
    return cfg


def parse_args_notebook_safe():
    ap = argparse.ArgumentParser(description="Generate centered pixel mushrooms.", add_help=True)
    ap.add_argument("--cap_color")
    ap.add_argument("--cap_shadow_color")
    ap.add_argument("--spot_color")
    ap.add_argument("--stem_color")
    ap.add_argument("--stem_shadow_color")
    ap.add_argument("--ring_color")
    ap.add_argument("--outline_color")
    ap.add_argument("--eye_color")

    ap.add_argument("--grid_w", type=int); ap.add_argument("--grid_h", type=int)
    ap.add_argument("--centered", type=lambda s: s.lower() in ("1","true","yes"))
    ap.add_argument("--offset_x", type=int); ap.add_argument("--offset_y", type=int)
    ap.add_argument("--cap_w", type=int); ap.add_argument("--cap_h", type=int)
    ap.add_argument("--cap_flat", type=int); ap.add_argument("--cap_round", type=float)
    ap.add_argument("--stem_w", type=int); ap.add_argument("--stem_h", type=int)
    ap.add_argument("--ring_thickness", type=int); ap.add_argument("--spot_radius", type=int)
    ap.add_argument("--add_center_spot", type=lambda s: s.lower() in ("1","true","yes"))
    ap.add_argument("--add_side_spots", type=lambda s: s.lower() in ("1","true","yes"))
    ap.add_argument("--add_eyes", type=lambda s: s.lower() in ("1","true","yes"))
    ap.add_argument("--eye_height_from_top_of_stem", type=int); ap.add_argument("--eye_gap", type=int)
    ap.add_argument("--palette_name", choices=list(PALETTES.keys()))
    ap.add_argument("--add_outlines", type=lambda s: s.lower() in ("1","true","yes"))
    ap.add_argument("--shadow_on_bottom_of_cap", type=lambda s: s.lower() in ("1","true","yes"))
    ap.add_argument("--scale", type=int)
    ap.add_argument("--out_prefix", default="mushroom"); ap.add_argument("--count", type=int, default=1)
    args, _unknown = ap.parse_known_args(); return args

def main():
    args = parse_args_notebook_safe()
    params = build_from_args(args)
    os.makedirs("out_mushrooms", exist_ok=True)
    for i in range(max(1, args.count)):
        sprite = draw_mushroom(params)
        name = f"{args.out_prefix}_{i+1}.png" if args.count > 1 else f"{args.out_prefix}.png"
        save_upscaled(sprite, params.scale, os.path.join("out_mushrooms", name))
    print("Saved to ./out_mushrooms")

# Export for importers
__all__ = ["MushroomParams", "draw_mushroom", "save_upscaled", "main"]

if __name__ == "__main__":
    main()
