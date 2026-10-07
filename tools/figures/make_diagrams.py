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
TEAL_FILL, TEAL_STROKE = "#d4ece9", "#4f9a92"   # entropy coding / bitstream
TEAL_SOFT = "#eef7f6"
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
G_A = ["Conv 5&#215;5, &#8595;2", "GDN", "RBB", "Attention", "Conv 5&#215;5, &#8595;2", "GDN", "RBB",
       "Conv 5&#215;5, &#8595;2", "Attention"]                                   # top to bottom = data order
G_S = ["Deconv 5&#215;5, &#8593;2", "IGDN", "RBB", "Attention", "Deconv 5&#215;5, &#8593;2", "IGDN", "RBB",
       "Deconv 5&#215;5, &#8593;2", "Attention"]                                 # bottom to top = data order


def stack(s, x, y, w, title, layers):
    s.rect(x, y, w, 290, BLUE_FILL, BLUE_STROKE)
    s.text(x + w / 2, y + 20, title, 12, weight="bold")
    for i, name in enumerate(layers):
        s.box(x + 16, y + 32 + i * 28, w - 32, 22, [(name, 11, "normal")], "#ffffff", BLUE_STROKE)


def coder_row(s, y, prior_label):
    s.box(204, y - 12, 24, 24, [("Q", 12, "normal")])
    s.box(240, y - 12, 40, 24, [("AE", 11, "normal")], TEAL_FILL, TEAL_STROKE)
    s.rect(292, y - 9, 48, 18, TEAL_SOFT, TEAL_STROKE, rx=2)
    s.text(316, y + 4, "0110&#8230;", 10, family="Menlo, Consolas, monospace")
    s.box(352, y - 12, 40, 24, [("AD", 11, "normal")], TEAL_FILL, TEAL_STROKE)
    s.arrow([(228, y), (238, y)])
    s.line([(280, y), (292, y)])
    s.arrow([(340, y), (350, y)])
    if prior_label:
        s.text(316, y + 26, prior_label, 10.5, fill=MUTED)


def codec():
    s = Svg(680, 640, "HyRES residual codec: scale hyperprior with checkerboard context model")
    # Main transforms
    s.text(100, 22, var("r") + "  (3 &#215; " + var("H") + " &#215; " + var("W") + ")", 12)
    s.arrow([(100, 28), (100, 46)])
    stack(s, 24, 48, 152, "Analysis " + var("g", "a"), G_A)
    s.text(580, 22, var("r", hat=True), 13)
    s.arrow([(580, 48), (580, 30)])
    stack(s, 504, 48, 152, "Synthesis " + var("g", "s"), G_S)

    # Latent y: quantize, arithmetic-code, decode
    s.line([(100, 338), (100, 372)])
    s.text(108, 358, var("y") + "  (192 &#215; " + var("H") + "/8 &#215; " + var("W") + "/8)", 11, anchor="start")
    s.arrow([(100, 372), (202, 372)])
    s.dot(100, 372)
    coder_row(s, 372, None)
    s.arrow([(392, 372), (580, 372), (580, 340)])
    s.text(412, 366, var("y", hat=True), 12)
    s.dot(460, 372)

    # Entropy parameters from hyperprior + checkerboard context of decoded anchors
    s.box(216, 404, 180, 40, [("Entropy parameters", 12, "normal"), ("3&#215; Conv 1&#215;1", 10.5, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.box(420, 404, 80, 40, [("Context", 12, "normal"), ("masked 5&#215;5", 10.5, "normal")], BLUE_FILL, BLUE_STROKE)
    s.arrow([(460, 372), (460, 402)])
    s.text(466, 392, "anchors", 10.5, anchor="start", fill=MUTED)
    s.arrow([(420, 424), (398, 424)])
    s.arrow([(260, 404), (260, 386)])
    s.arrow([(372, 404), (372, 386)])
    s.text(316, 398, "&#956;, &#963;", 11, family=SERIF, italic=True)

    # Hyperprior: h_a -> z -> factorized coding -> h_s -> entropy parameters
    s.arrow([(100, 372), (100, 470)])
    s.box(24, 472, 152, 48, [("Hyper analysis " + var("h", "a"), 12, "bold"), ("Conv 3&#215;3, 2&#215; Conv &#8595;2", 10.5, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.line([(100, 520), (100, 560)])
    s.text(108, 544, var("z") + "  (128 &#215; " + var("H") + "/32 &#215; " + var("W") + "/32)", 11, anchor="start")
    s.arrow([(100, 560), (202, 560)])
    coder_row(s, 560, "factorized prior")
    s.arrow([(392, 560), (460, 560), (460, 522)])
    s.box(392, 472, 136, 48, [("Hyper synthesis " + var("h", "s"), 12, "bold"), ("2&#215; Deconv &#8593;2, Conv 3&#215;3", 10.5, "normal")],
          BLUE_FILL, BLUE_STROKE)
    s.arrow([(460, 472), (460, 458), (306, 458), (306, 446)])

    # Checkerboard decoding order
    ox, oy, c = 556, 392, 20
    for i in range(4):
        for j in range(4):
            anchor = (i + j) % 2 == 0
            s.rect(ox + j * (c + 4), oy + i * (c + 4), c, c, BLUE_STROKE if anchor else "#ffffff", BLUE_STROKE, rx=2)
            s.text(ox + j * (c + 4) + c / 2, oy + i * (c + 4) + 14, "1" if anchor else "2", 10.5,
                   fill="#ffffff" if anchor else INK)
    s.text(600, 510, "Decoding order:", 10.5, fill=MUTED)
    s.text(600, 524, "1 anchors, then 2", 10.5, fill=MUTED)

    legend(s, 626, [("blue", "Learned module"), ("teal", "Entropy coding"), ("white", "Quantizer Q")])
    s.text(652, 627, "RBB: residual bottleneck block", 10.5, anchor="end", fill=MUTED)
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
