"""Generate the two method diagrams as standalone SVG, in the visual language of the CARI figure.

  Fig. 1  docs/static/fig/hyres_pipeline.svg   JPEG base layer + learned residual layer
  Fig. 2  docs/static/fig/hyres_codec.svg      residual codec (hyperprior + checkerboard context)

Style (taken from the CARI mechanism figure):
  - two accents only: blue #1677ff = learned stream, ochre #b7791f = JPEG / entropy-coded stream
  - accents colour the DATA (1.7px frames, bold headings), not containers; no group boxes
  - dark navy #172033 titles, slate #475467 arrows and secondary text, #d9e0e8 neutral frames
  - soft curved slate arrows with small solid heads, large rounded cards for modules
  - blue and ochre fills differ in lightness so they stay distinct in grayscale print
Layer lists in Fig. 2 mirror models/checkerboard.py; update both together.

The thumbnails in Fig. 1 are real Phase 1 outputs. Usage (from the repo root):
    python tools/figures/make_diagrams.py --eval-dir output/phase1 [--image kodim23]
"""
import argparse
import base64
import csv
import io
import math
import os
import sys

import numpy as np
from PIL import Image

OUT = os.path.join("docs", "static", "fig")

NAVY, SLATE, FRAME = "#172033", "#475467", "#d9e0e8"
BLUE, BLUE_FILL = "#1677ff", "#eaf3ff"
OCHRE, OCHRE_FILL = "#b7791f", "#f6e3bd"   # darker fill than blue's, so the two differ in grayscale
SANS = "Helvetica, Arial, sans-serif"
SERIF = "'Times New Roman', Times, serif"
SW = 1.7


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts, self.title = w, h, [], title

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="#ffffff", stroke=SLATE, rx=8, sw=SW, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=12, anchor="middle", weight="normal", fill=NAVY, family=SANS, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{st}>{s}</text>')

    def vtext(self, cx, cy, s, size=11, fill=NAVY, weight="normal"):
        """Text rotated 90 degrees counter-clockwise, centered on (cx, cy)."""
        self.add(f'<text x="{cx}" y="{cy}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                 f'text-anchor="middle" dominant-baseline="central" transform="rotate(-90 {cx} {cy})">{s}</text>')

    def card(self, x, y, w, h, title, sub=None, kind="blue", sub2=None, size=14):
        """Rounded module card: tinted fill, accent outline, navy title, slate sub-lines."""
        fill, stroke = (BLUE_FILL, BLUE) if kind == "blue" else (OCHRE_FILL, OCHRE)
        self.rect(x, y, w, h, fill, stroke, rx=12)
        cx = x + w / 2
        lines = [(title, size, "bold", NAVY)] + [(t, 11.5, "normal", SLATE) for t in (sub, sub2) if t]
        total = sum(sz + 6 for _, sz, _, _ in lines) - 6
        yy = y + (h - total) / 2
        for t, sz, wt, col in lines:
            self.text(cx, yy + sz * 0.82, t, sz, weight=wt, fill=col)
            yy += sz + 6

    def box(self, x, y, w, h, lines, fill="#ffffff", stroke=SLATE, rx=6, sw=1.5, dash=None):
        """Small tile with 1-2 centered lines: (text, size, weight)."""
        self.rect(x, y, w, h, fill, stroke, rx=rx, sw=sw, dash=dash)
        cx = x + w / 2
        if len(lines) == 1:
            t, s, wt = lines[0]
            self.text(cx, y + h / 2 + s * 0.36, t, s, weight=wt)
        else:
            (t1, s1, w1), (t2, s2, w2) = lines
            gap = 3
            top = y + (h - (s1 + s2 + gap)) / 2
            self.text(cx, top + s1 * 0.82, t1, s1, weight=w1)
            self.text(cx, top + s1 + gap + s2 * 0.82, t2, s2, weight=w2, fill=SLATE)

    # ---- arrows: slate, 1.7px, round caps, small solid triangular heads
    def _head(self, tip, d, color):
        base = (tip[0] - 8 * d[0], tip[1] - 8 * d[1])
        px, py = -d[1], d[0]
        pts = [tip, (base[0] + 3.8 * px, base[1] + 3.8 * py), (base[0] - 3.8 * px, base[1] - 3.8 * py)]
        self.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                 f'" fill="{color}" stroke="{color}" stroke-width="1" stroke-linejoin="round"/>')

    @staticmethod
    def _unit(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    def arrow(self, pts, color=SLATE, dash=None, head=True):
        """Polyline arrow with a head at the last point."""
        pts = [tuple(p) for p in pts]
        d = self._unit(pts[-2], pts[-1])
        end = (pts[-1][0] - 6 * d[0], pts[-1][1] - 6 * d[1]) if head else pts[-1]
        path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts[:-1] + [end])
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{SW}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{da}/>')
        if head:
            self._head(pts[-1], d, color)

    def line(self, pts, color=SLATE):
        self.arrow(pts, color=color, head=False)

    def curve(self, p0, c1, c2, p3, color=SLATE):
        """Cubic Bezier arrow; the head follows the end tangent."""
        d = self._unit(c2, p3)
        end = (p3[0] - 6 * d[0], p3[1] - 6 * d[1])
        self.add(f'<path d="M{p0[0]} {p0[1]} C{c1[0]} {c1[1]} {c2[0]} {c2[1]} {end[0]:.1f} {end[1]:.1f}" fill="none" '
                 f'stroke="{color}" stroke-width="{SW}" stroke-linecap="round"/>')
        self._head(p3, d, color)

    def qcurve(self, p0, p2, bow, color=SLATE):
        """Gentle quadratic arrow between two points; bow bends it perpendicular to the chord."""
        mx, my = (p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2
        c = (mx, my + bow)
        d = self._unit(c, p2)
        end = (p2[0] - 6 * d[0], p2[1] - 6 * d[1])
        self.add(f'<path d="M{p0[0]} {p0[1]} Q{c[0]:.1f} {c[1]:.1f} {end[0]:.1f} {end[1]:.1f}" fill="none" '
                 f'stroke="{color}" stroke-width="{SW}" stroke-linecap="round"/>')
        self._head(p2, d, color)

    def dot(self, x, y):
        self.add(f'<circle cx="{x}" cy="{y}" r="2.8" fill="{SLATE}"/>')

    def plus(self, cx, cy, r=13):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{SLATE}" stroke-width="{SW}"/>')
        self.add(f'<path d="M{cx - 6} {cy}H{cx + 6}M{cx} {cy - 6}V{cy + 6}" stroke="{SLATE}" stroke-width="{SW}" stroke-linecap="round"/>')

    def image(self, x, y, w, h, data_uri, frame=FRAME):
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{data_uri}" preserveAspectRatio="none"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{frame}" stroke-width="{SW}"/>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img" aria-label="{self.title}">\n'
                f'<title>{self.title}</title>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as f:
            f.write(head + "\n".join(self.parts) + "\n</svg>\n")
        print(f"Saved {path}")


def var(base, sub="", hat=False):
    """Math variable in serif italic, e.g. var('x', 'J') -> x_J; hat adds a circumflex."""
    h = "&#770;" if hat else ""
    s = f'<tspan font-family="{SERIF}" font-style="italic">{base}{h}</tspan>'
    if sub:
        s += (f'<tspan font-family="{SERIF}" font-style="italic" font-size="75%" dy="3">{sub}</tspan>'
              '<tspan dy="-3"></tspan>')
    return s


def thumb_uri(arr, w=336, h=224):
    img = Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8)).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load(path):
    if not os.path.isfile(path):
        sys.exit(f"error: missing {path}; run scripts/evaluate.sh first (see module docstring)")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def image_stats(eval_dir, image):
    """Measured numbers for one image from evaluate.sh's metrics.csv, so the figure matches its thumbnails."""
    path = os.path.join(eval_dir, "metrics.csv")
    if not os.path.isfile(path):
        sys.exit(f"error: missing {path}; run scripts/evaluate.sh first")
    with open(path) as f:
        for row in csv.DictReader(f):
            if os.path.splitext(row["filename"])[0] == image:
                total, jpeg = float(row["total_bpp"]), float(row["jpeg_bpp"])
                return {"total": total, "jpeg": jpeg, "residual": total - jpeg, "psnr": float(row["psnr"])}
    sys.exit(f"error: {image} not found in {path}")


# ---------------------------------------------------------------- Fig. 1
def pipeline(eval_dir, image, kodak):
    orig = load(os.path.join(kodak, f"{image}.png"))
    jpeg = load(os.path.join(eval_dir, f"{image}_jpeg.png"))
    recon = load(os.path.join(eval_dir, f"{image}_recon.png"))
    st = image_stats(eval_dir, image)
    t = {"x": thumb_uri(orig), "xj": thumb_uri(jpeg), "xhat": thumb_uri(recon),
         # Residual shown amplified 2x around mid-gray so it is visible.
         "rhat": thumb_uri(127.5 + 2 * (recon - jpeg))}
    s = Svg(1000, 410, "HyRES pipeline: a JPEG base layer plus a learned residual layer, summed at the decoder")
    TW, TH = 168, 112

    # Column 1: input
    s.text(100, 136, "Input image  " + var("x"), 14, weight="bold")
    s.image(16, 148, TW, TH, t["x"])
    s.text(100, 282, f"Kodak {image}", 11.5, fill=SLATE)

    # Column 2: the two codecs, drawn as cards (like the shared model in the CARI figure)
    s.card(250, 68, 140, 80, "JPEG codec", "standard, quality 1", kind="ochre")
    s.card(250, 260, 140, 80, "Residual codec", "learned, 10.1M params", kind="blue",
           sub2="input  " + var("r") + " = " + var("x") + " &#8722; " + var("x", "J"))
    s.curve((188, 204), (219, 204), (219, 108), (248, 108))
    s.curve((188, 204), (219, 204), (219, 300), (248, 300))
    s.arrow([(320, 150), (320, 258)])
    s.text(330, 210, var("x", "J") + ", subtracted from " + var("x"), 11.5, anchor="start", fill=SLATE)

    # Column 3: the two layers; arrow labels carry the measured bitrate of each stream
    s.text(554, 42, "JPEG layer  " + var("x", "J"), 14, weight="bold", fill=OCHRE)
    s.image(470, 52, TW, TH, t["xj"], OCHRE)
    s.text(554, 184, "decoded JPEG, quality 1", 11.5, fill=SLATE)
    s.qcurve((392, 108), (468, 108), 5)
    s.text(430, 97, f"{st['jpeg']:.2f} bpp", 11, weight="bold", fill=OCHRE)

    s.image(470, 244, TW, TH, t["rhat"], BLUE)
    s.text(554, 380, "Residual layer  " + var("r", hat=True), 14, weight="bold", fill=BLUE)
    s.text(554, 398, "decoded residual (shown 2&#215;)", 11.5, fill=SLATE)
    s.qcurve((392, 300), (468, 300), -5)
    s.text(430, 289, f"{st['residual']:.2f} bpp", 11, weight="bold", fill=BLUE)

    # Column 4: sum and output
    s.curve((640, 108), (716, 108), (716, 150), (716, 191))
    s.curve((640, 300), (716, 300), (716, 258), (716, 217))
    s.plus(716, 204)
    s.arrow([(730, 204), (798, 204)])
    s.text(884, 136, "Output  " + var("x", hat=True) + " = " + var("x", "J") + " + " + var("r", hat=True),
           14, weight="bold")
    s.image(800, 148, TW, TH, t["xhat"], NAVY)
    s.text(884, 282, f"{st['total']:.2f} bpp in total, {st['psnr']:.1f} dB", 11.5, fill=SLATE)
    s.save(os.path.join(OUT, "hyres_pipeline.svg"))


# ---------------------------------------------------------------- Fig. 2
# Layer order follows models/checkerboard.py (data order). All convolutions are 5x5 except where noted.
G_A = [("Conv &#8595;2", "H/2 &#215; W/2 &#215; 128"), ("GDN", None), ("RBB", None), ("Attention", None),
       ("Conv &#8595;2", "H/4 &#215; W/4 &#215; 128"), ("GDN", None), ("RBB", None),
       ("Conv &#8595;2", "H/8 &#215; W/8 &#215; 192"), ("Attention", None)]
G_S = [("Attention", None), ("Deconv &#8593;2", "H/4 &#215; W/4 &#215; 128"), ("RBB", None), ("IGDN", None),
       ("Deconv &#8593;2", "H/2 &#215; W/2 &#215; 128"), ("Attention", None), ("RBB", None), ("IGDN", None),
       ("Deconv &#8593;2", "H &#215; W &#215; 3")]
TILE_W, TILE_H, PITCH, X0 = 28, 84, 34, 84


def tiles(s, layers, cy, reverse):
    """Draw a row of layer tiles; reverse=True lays data order right-to-left (synthesis)."""
    for i, (name, shape) in enumerate(layers):
        x = X0 + ((len(layers) - 1 - i) if reverse else i) * PITCH
        s.rect(x, cy - TILE_H / 2, TILE_W, TILE_H, BLUE_FILL, BLUE, rx=6, sw=1.5)
        s.vtext(x + TILE_W / 2, cy, name, 11)
        if shape:
            s.text(x + TILE_W / 2, cy + TILE_H / 2 + 14, shape, 9.5, fill=SLATE)


def coder(s, cx, y_top):
    """Q, AE, bits, AD stacked vertically; returns the y of the AD tile's bottom edge."""
    s.box(cx - 12, y_top, 24, 22, [("Q", 11, "normal")])
    s.arrow([(cx, y_top + 22), (cx, y_top + 36)])
    s.rect(cx - 20, y_top + 38, 40, 22, OCHRE_FILL, OCHRE, rx=11, sw=1.5)        # pills = entropy coding
    s.text(cx, y_top + 53, "AE", 11)
    s.arrow([(cx, y_top + 60), (cx, y_top + 72)])
    for k, bit in enumerate([1, 0, 0, 1, 1]):
        s.rect(cx - 18.75 + k * 7.5, y_top + 74, 7, 10, NAVY if bit else "#ffffff", NAVY, rx=0, sw=0.8)
    s.arrow([(cx, y_top + 84), (cx, y_top + 94)])
    s.rect(cx - 20, y_top + 96, 40, 22, OCHRE_FILL, OCHRE, rx=11, sw=1.5)
    s.text(cx, y_top + 111, "AD", 11)
    return y_top + 118


def codec():
    s = Svg(968, 356, "HyRES residual codec: main path and entropy model")
    s.text(16, 20, "(a)  Residual codec", 13, anchor="start", weight="bold")
    s.text(680, 20, "(b)  Entropy model", 13, anchor="start", weight="bold")

    # Main path: analysis on top (left to right), synthesis mirrored below (right to left). No group boxes.
    s.text(84, 48, "Analysis transform  " + var("g", "a"), 12.5, anchor="start", weight="bold")
    s.text(290, 48, "runs in the encoder", 11, anchor="start", fill=SLATE)
    s.text(84, 188, "Synthesis transform  " + var("g", "s"), 12.5, anchor="start", weight="bold")
    s.text(290, 188, "runs in the decoder", 11, anchor="start", fill=SLATE)
    tiles(s, G_A, 100, reverse=False)
    tiles(s, G_S, 244, reverse=True)
    s.text(30, 96, var("r"), 14)
    s.arrow([(40, 100), (82, 100)])
    s.arrow([(82, 244), (24, 244)])
    s.text(30, 238, var("r", hat=True), 14, anchor="end")

    # Quantize + arithmetic-code y (pills are entropy coding); the decoder side mirrors it.
    s.arrow([(384, 100), (526, 100), (526, 112)])
    s.dot(526, 100)
    s.text(468, 93, var("y"), 13)
    bottom = coder(s, 526, 114)
    s.arrow([(526, bottom), (526, 244), (386, 244)])
    s.dot(526, 244)
    s.text(468, 237, var("y", hat=True), 13)

    # The entropy model is one card here; panel (b) opens it.
    s.rect(584, 120, 70, 156, BLUE_FILL, BLUE, rx=12)
    s.vtext(612, 198, "Entropy model", 13, weight="bold")
    s.vtext(634, 198, "used by encoder and decoder", 10.5, fill=SLATE)
    s.arrow([(526, 100), (619, 100), (619, 118)])
    s.arrow([(584, 163), (550, 163)])
    s.arrow([(584, 219), (550, 219)])
    s.text(567, 156, "&#956;, &#963;", 11, family=SERIF, italic=True)
    s.text(567, 212, "&#956;, &#963;", 11, family=SERIF, italic=True)
    s.arrow([(526, 244), (526, 296), (619, 296), (619, 278)], dash="4 3")
    s.text(572, 312, "decoded anchors", 10.5, fill=SLATE)

    s.line([(668, 32), (668, 312)], color=FRAME)

    # (b) Entropy model detail: hyperprior column + context model.
    s.text(742, 50, var("y"), 13)
    s.arrow([(742, 56), (742, 68)])
    s.card(688, 70, 108, 38, "Hyper analysis " + var("h", "a"), kind="blue", size=12)
    s.arrow([(742, 108), (742, 122)])
    s.text(750, 118, var("z") + "  H/32 &#215; W/32", 10.5, anchor="start")
    s.card(688, 124, 108, 40, "Q, AE / AD", "factorized prior", kind="ochre", size=12)
    s.arrow([(742, 164), (742, 178)])
    s.card(688, 180, 108, 38, "Hyper synthesis " + var("h", "s"), kind="blue", size=12)
    s.arrow([(742, 218), (742, 232)])
    s.card(688, 234, 108, 42, "Entropy params", "3&#215; Conv 1&#215;1", kind="blue", size=12)
    s.arrow([(742, 276), (742, 292)])
    s.text(742, 306, "&#956;, &#963; per latent", 11, family=SERIF, italic=True)

    s.card(832, 234, 112, 42, "Context model", "masked Conv 5&#215;5", kind="blue", size=12)
    s.arrow([(832, 255), (798, 255)])
    s.text(888, 212, "decoded anchors", 10.5, fill=SLATE)
    s.arrow([(888, 217), (888, 232)])
    s.text(888, 292, "pass 2 only; pass 1 uses", 10.5, fill=SLATE)
    s.text(888, 305, "the hyperprior alone", 10.5, fill=SLATE)

    # Checkerboard order: anchors (filled) are coded first; the rest use them as context.
    ox, oy, c = 850, 62, 17
    for i in range(4):
        for j in range(4):
            a = (i + j) % 2 == 0
            s.rect(ox + j * (c + 4), oy + i * (c + 4), c, c, BLUE if a else "#ffffff", BLUE, rx=3, sw=1.4)
            s.text(ox + j * (c + 4) + c / 2, oy + i * (c + 4) + 12.5, "1" if a else "2", 10, fill="#ffffff" if a else NAVY)
    s.text(888, 178, "Checkerboard: pass 1 codes", 10.5, fill=SLATE)
    s.text(888, 191, "anchors, pass 2 the rest", 10.5, fill=SLATE)

    # Key + abbreviations
    s.rect(16, 332, 18, 12, BLUE_FILL, BLUE, rx=4, sw=1.4); s.text(40, 342, "Learned module", 11, anchor="start", fill=SLATE)
    s.rect(150, 332, 26, 12, OCHRE_FILL, OCHRE, rx=6, sw=1.4); s.text(182, 342, "Entropy coding (AE/AD)", 11, anchor="start", fill=SLATE)
    s.rect(330, 332, 18, 12, "#ffffff", SLATE, rx=3, sw=1.4); s.text(354, 342, "Quantizer", 11, anchor="start", fill=SLATE)
    s.text(952, 342, "GDN: generalized divisive normalization.  RBB: residual bottleneck block.  Conv/Deconv: 5&#215;5, stride 2",
           10, anchor="end", fill=SLATE)
    s.save(os.path.join(OUT, "hyres_codec.svg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="output/phase1")
    ap.add_argument("--image", default="kodim23")
    ap.add_argument("--kodak", default="data/test")
    args = ap.parse_args()
    pipeline(args.eval_dir, args.image, args.kodak)
    codec()


if __name__ == "__main__":
    main()
