"""Generate the two publication diagrams as standalone SVG (draw.io-style flat design).

  Fig. 1  docs/static/fig/hyres_pipeline.svg   JPEG base layer + learned residual layer
  Fig. 2  docs/static/fig/hyres_codec.svg      residual codec (hyperprior + checkerboard context)

Style rules: white background, #333 text and outlines, 1px strokes, 4px corner
radius, orthogonal arrows, Helvetica/Arial (math variables in serif italic),
exactly two accents: blue = learned modules, teal = entropy coding / bitstream.
Layer lists in Fig. 2 mirror models/checkerboard.py; update both together.

The thumbnails in Fig. 1 are real Phase 1 outputs. Usage (from the repo root):
    python tools/figures/make_diagrams.py --eval-dir output/phase1 [--image kodim23]
"""
import argparse
import base64
import io
import os
import sys

import numpy as np
from PIL import Image

OUT = os.path.join("docs", "static", "fig")

INK = "#333333"
MUTED = "#666666"
GROUP_FILL, GROUP_STROKE = "#f7f7f7", "#9e9e9e"
BLUE_FILL, BLUE_STROKE = "#dae8fc", "#6c8ebf"   # learned modules
TEAL_FILL, TEAL_STROKE = "#b8dcd6", "#2f7d74"   # entropy coding / bitstream (darker than blue in grayscale)
TEAL_SOFT = "#e3f1ef"
SANS = "Helvetica, Arial, sans-serif"
SERIF = "'Times New Roman', Times, serif"


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.title = title

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="#ffffff", stroke=INK, dash=None, rx=4, sw=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=12, anchor="middle", weight="normal", fill=INK, family=SANS, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{st}>{s}</text>')

    def vtext(self, cx, cy, s, size=11, fill=INK):
        """Text rotated 90 degrees counter-clockwise, centered on (cx, cy)."""
        self.add(f'<text x="{cx}" y="{cy}" font-family="{SANS}" font-size="{size}" fill="{fill}" text-anchor="middle" '
                 f'dominant-baseline="central" transform="rotate(-90 {cx} {cy})">{s}</text>')

    def box(self, x, y, w, h, lines, fill="#ffffff", stroke=INK, dash=None):
        """Box with 1-2 centered lines: (text, size, weight)."""
        self.rect(x, y, w, h, fill, stroke, dash)
        cx = x + w / 2
        if len(lines) == 1:
            t, s, wt = lines[0]
            self.text(cx, y + h / 2 + s * 0.36, t, s, weight=wt)
        else:
            (t1, s1, w1), (t2, s2, w2) = lines
            gap = 3
            top = y + (h - (s1 + s2 + gap)) / 2
            self.text(cx, top + s1 * 0.82, t1, s1, weight=w1)
            self.text(cx, top + s1 + gap + s2 * 0.82, t2, s2, weight=w2, fill=MUTED)

    def arrow(self, pts, dash=None):
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1" marker-end="url(#ah)"{da}/>')

    def line(self, pts):
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        self.add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1"/>')

    def dot(self, x, y):
        self.add(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{INK}"/>')

    def op(self, cx, cy, sym):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="10" fill="#ffffff" stroke="{INK}" stroke-width="1"/>')
        if sym == "+":
            self.line([(cx - 5, cy), (cx + 5, cy)])
            self.line([(cx, cy - 5), (cx, cy + 5)])
        else:
            self.line([(cx - 5, cy), (cx + 5, cy)])

    def image(self, x, y, w, h, data_uri):
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{data_uri}" preserveAspectRatio="none"/>')
        self.rect(x, y, w, h, fill="none", stroke=INK, rx=0)

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img" aria-label="{self.title}">\n'
                f'<title>{self.title}</title>\n'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" '
                f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0 1 L10 5 L0 9 z" fill="{INK}"/></marker></defs>\n'
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


def thumb_uri(arr, w=144, h=96):
    img = Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8)).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load(path):
    if not os.path.isfile(path):
        sys.exit(f"error: missing {path}; run scripts/evaluate.sh first (see module docstring)")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def legend(s, y, items):
    x = 28
    for kind, label in items:
        if kind == "blue":
            s.rect(x, y - 9, 18, 12, BLUE_FILL, BLUE_STROKE, rx=2)
        elif kind == "teal":
            s.rect(x, y - 9, 18, 12, TEAL_FILL, TEAL_STROKE, rx=2)
        elif kind == "dash":
            s.rect(x, y - 9, 18, 12, "#ffffff", BLUE_STROKE, dash="3 2", rx=2)
        else:
            s.rect(x, y - 9, 18, 12, "#ffffff", INK, rx=2)
        s.text(x + 24, y + 1, label, 11, anchor="start")
        x += 24 + len(label) * 6.1 + 22


# ---------------------------------------------------------------- Fig. 1
def pipeline(eval_dir, image, kodak):
    orig = load(os.path.join(kodak, f"{image}.png"))
    jpeg = load(os.path.join(eval_dir, f"{image}_jpeg.png"))
    recon = load(os.path.join(eval_dir, f"{image}_recon.png"))
    t = {
        "x": thumb_uri(orig), "xj": thumb_uri(jpeg), "xhat": thumb_uri(recon),
        # Residuals amplified 2x around mid-gray so they are visible in print.
        "r": thumb_uri(127.5 + 2 * (orig - jpeg)), "rhat": thumb_uri(127.5 + 2 * (recon - jpeg)),
    }
    s = Svg(656, 500, "HyRES pipeline: JPEG base layer plus learned residual layer")
    TW, TH = 72, 48

    # Encoder group
    s.rect(16, 16, 624, 168, GROUP_FILL, GROUP_STROKE, dash="4 3")
    s.text(28, 36, "Encoder", 12, anchor="start", weight="bold")
    s.image(28, 84, TW, TH, t["x"]); s.text(64, 150, var("x"), 13)
    s.box(124, 88, 96, 40, [("JPEG encoder", 12, "normal"), ("quality 1", 10.5, "normal")])
    s.box(244, 88, 96, 40, [("JPEG decoder", 12, "normal")])
    s.op(372, 108, "-")
    s.image(400, 84, TW, TH, t["r"]); s.text(436, 150, var("r") + " = " + var("x") + " &#8722; " + var("x", "J"), 13)
    s.box(496, 88, 128, 40, [("Residual encoder", 12, "normal"), ("analysis + hyperprior", 10.5, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.arrow([(100, 108), (122, 108)])
    s.arrow([(220, 108), (242, 108)])
    s.arrow([(340, 108), (360, 108)])
    s.text(351, 101, var("x", "J"), 11)
    s.arrow([(382, 108), (398, 108)])
    s.arrow([(472, 108), (494, 108)])
    s.arrow([(64, 84), (64, 60), (372, 60), (372, 96)])          # x bypasses JPEG to the subtractor
    s.arrow([(172, 128), (172, 206)])
    s.arrow([(560, 128), (560, 206)])

    # Bitstream
    s.text(28, 224, "Bitstream", 12, anchor="start", weight="bold")
    s.text(28, 239, "1.56 bpp", 10.5, anchor="start", fill=MUTED)
    s.rect(116, 204, 516, 46, TEAL_SOFT, TEAL_STROKE)
    s.box(124, 210, 96, 34, [("JPEG &#183; 0.20 bpp", 11, "normal")], TEAL_FILL, TEAL_STROKE)
    s.box(400, 210, 224, 34, [("Residual " + var("y") + ", " + var("z") + " &#183; 1.36 bpp", 11, "normal")],
          TEAL_FILL, TEAL_STROKE)

    # Decoder group
    s.rect(16, 270, 624, 192, GROUP_FILL, GROUP_STROKE, dash="4 3")
    s.text(28, 290, "Decoder", 12, anchor="start", weight="bold")
    s.arrow([(172, 244), (172, 322)])
    s.arrow([(560, 244), (560, 322)])
    s.box(124, 324, 96, 40, [("JPEG decoder", 12, "normal")])
    s.image(244, 320, TW, TH, t["xj"]); s.text(280, 384, var("x", "J"), 13)
    s.op(372, 344, "+")
    s.image(400, 320, TW, TH, t["rhat"]); s.text(436, 384, var("r", hat=True), 13)
    s.box(496, 324, 128, 40, [("Residual decoder", 12, "normal"), ("hyperprior + synthesis", 10.5, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.arrow([(220, 344), (242, 344)])
    s.arrow([(316, 344), (360, 344)])
    s.arrow([(400, 344), (384, 344)])
    s.arrow([(496, 344), (474, 344)])
    s.arrow([(372, 356), (372, 398)])
    s.box(316, 400, 112, 40, [("Refinement", 12, "normal"), ("optional", 10.5, "normal")], "#ffffff", BLUE_STROKE, dash="4 3")
    s.arrow([(428, 420), (450, 420)])
    s.image(452, 396, TW, TH, t["xhat"]); s.text(538, 425, var("x", hat=True) + " (output)", 13, anchor="start")

    legend(s, 486, [("white", "Standard JPEG"), ("blue", "Learned"), ("teal", "Entropy-coded data"), ("dash", "Optional")])
    s.save(os.path.join(OUT, "hyres_pipeline.svg"))


# ---------------------------------------------------------------- Fig. 2
# Layer order follows models/checkerboard.py (data order). All convolutions are 5x5 except where noted.
G_A = [("Conv &#8595;2", "H/2 &#215; W/2 &#215; 128"), ("GDN", None), ("RBB", None), ("Attention", None),
       ("Conv &#8595;2", "H/4 &#215; W/4 &#215; 128"), ("GDN", None), ("RBB", None),
       ("Conv &#8595;2", "H/8 &#215; W/8 &#215; 192"), ("Attention", None)]
G_S = [("Attention", None), ("Deconv &#8593;2", "H/4 &#215; W/4"), ("RBB", None), ("IGDN", None),
       ("Deconv &#8593;2", "H/2 &#215; W/2"), ("Attention", None), ("RBB", None), ("IGDN", None),
       ("Deconv &#8593;2", "H &#215; W &#215; 3")]
TILE_W, TILE_H, PITCH, X0 = 28, 84, 34, 84


def tiles(s, layers, cy, reverse):
    """Draw a row of layer tiles; reverse=True lays data order right-to-left (synthesis)."""
    for i, (name, shape) in enumerate(layers):
        x = X0 + ((len(layers) - 1 - i) if reverse else i) * PITCH
        s.rect(x, cy - TILE_H / 2, TILE_W, TILE_H, BLUE_FILL, BLUE_STROKE)
        s.vtext(x + TILE_W / 2, cy, name, 11)
        if shape:
            s.text(x + TILE_W / 2, cy + TILE_H / 2 + 13, shape, 9.5, fill=MUTED)


def codec():
    s = Svg(968, 356, "HyRES residual codec: main path and entropy model")
    s.text(16, 20, "(a) Residual codec", 12, anchor="start", weight="bold")
    s.text(680, 20, "(b) Entropy model", 12, anchor="start", weight="bold")

    # Main path: analysis on top (left to right), synthesis mirrored below (right to left).
    s.rect(52, 32, 396, 128, GROUP_FILL, GROUP_STROKE, dash="4 3")
    s.text(62, 50, var("g", "a"), 13, anchor="start")
    s.rect(52, 172, 396, 136, GROUP_FILL, GROUP_STROKE, dash="4 3")
    s.text(62, 190, var("g", "s"), 13, anchor="start")
    tiles(s, G_A, 96, reverse=False)
    tiles(s, G_S, 236, reverse=True)
    s.text(30, 92, var("r"), 14)
    s.arrow([(38, 96), (82, 96)])
    s.arrow([(82, 236), (24, 236)])
    s.text(30, 230, var("r", hat=True), 14, anchor="end")

    # Quantize + arithmetic code y; the bitstream is drawn as bits.
    s.arrow([(384, 96), (526, 96), (526, 110)])
    s.text(468, 90, var("y"), 13)
    s.box(514, 112, 24, 22, [("Q", 11, "normal")])
    s.arrow([(526, 134), (526, 146)])
    s.box(506, 148, 40, 22, [("AE", 11, "normal")], TEAL_FILL, TEAL_STROKE)
    s.arrow([(526, 170), (526, 180)])
    for k, bit in enumerate([1, 0, 0, 1, 1]):
        s.rect(508 + k * 7.5, 182, 7, 10, INK if bit else "#ffffff", INK, rx=0, sw=0.8)
    s.arrow([(526, 192), (526, 202)])
    s.box(506, 204, 40, 22, [("AD", 11, "normal")], TEAL_FILL, TEAL_STROKE)
    s.arrow([(526, 226), (526, 236), (386, 236)])
    s.text(468, 230, var("y", hat=True), 13)

    # Entropy model as one box; it reads y (hyperprior) and decoded anchors (context).
    s.rect(584, 112, 64, 156, BLUE_FILL, BLUE_STROKE)
    s.vtext(616, 190, "Entropy model", 12)
    s.arrow([(526, 96), (616, 96), (616, 110)])
    s.arrow([(584, 159), (548, 159)])
    s.arrow([(584, 215), (548, 215)])
    s.text(566, 152, "&#956;, &#963;", 11, family=SERIF, italic=True)
    s.text(566, 208, "&#956;, &#963;", 11, family=SERIF, italic=True)
    s.add(f'<path d="M526 236 L526 286 L616 286 L616 270" fill="none" stroke="{INK}" stroke-width="1" '
          'stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    s.text(571, 300, "decoded anchors", 10, fill=MUTED)

    # Divider between panels
    s.line([(664, 32), (664, 308)])
    s.parts[-1] = s.parts[-1].replace(f'stroke="{INK}"', 'stroke="#cccccc"')

    # (b) Entropy model detail: hyperprior column + context model.
    s.text(740, 46, var("y"), 13)
    s.arrow([(740, 52), (740, 64)])
    s.box(686, 66, 108, 36, [("Hyper analysis " + var("h", "a"), 11.5, "normal")], BLUE_FILL, BLUE_STROKE)
    s.arrow([(740, 102), (740, 116)])
    s.text(748, 112, var("z"), 12, anchor="start")
    s.box(686, 118, 108, 36, [("Q, AE / AD", 11.5, "normal"), ("factorized prior", 10, "normal")], TEAL_FILL, TEAL_STROKE)
    s.arrow([(740, 154), (740, 168)])
    s.box(686, 170, 108, 36, [("Hyper synthesis " + var("h", "s"), 11.5, "normal")], BLUE_FILL, BLUE_STROKE)
    s.arrow([(740, 206), (740, 220)])
    s.box(686, 222, 108, 40, [("Entropy params", 11.5, "normal"), ("3&#215; Conv 1&#215;1", 10, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.arrow([(740, 262), (740, 280)])
    s.text(740, 294, "&#956;, &#963; per latent", 11, family=SERIF, italic=True)

    s.text(880, 196, "decoded anchors", 10, fill=MUTED)
    s.arrow([(880, 202), (880, 220)])
    s.box(828, 222, 104, 40, [("Context model", 11.5, "normal"), ("masked Conv 5&#215;5", 10, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.arrow([(828, 242), (796, 242)])

    # Checkerboard order: anchors decoded first, in parallel; the rest use them as context.
    ox, oy, c = 842, 66, 16
    for i in range(4):
        for j in range(4):
            a = (i + j) % 2 == 0
            s.rect(ox + j * (c + 4), oy + i * (c + 4), c, c, BLUE_STROKE if a else "#ffffff", BLUE_STROKE, rx=2)
            s.text(ox + j * (c + 4) + c / 2, oy + i * (c + 4) + 12, "1" if a else "2", 9.5,
                   fill="#ffffff" if a else INK)
    s.text(880, 164, "Checkerboard: pass 1 codes", 10, fill=MUTED)
    s.text(880, 177, "anchors, pass 2 the rest", 10, fill=MUTED)

    legend(s, 336, [("blue", "Learned"), ("teal", "Entropy coding"), ("white", "Quantizer")])
    s.text(952, 337, "RBB: residual bottleneck block. Conv/Deconv: 5&#215;5, stride 2", 10.5, anchor="end", fill=MUTED)
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
