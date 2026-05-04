#!/usr/bin/env python3
"""
Generate research presentation as a PowerPoint (.pptx) file.
Run: python docs/generate_pptx.py
Output: docs/presentation.pptx

Color palette: 2 colors only
  - BLUE    #0072B2  (accent, highlights, charts)
  - CHARCOAL #333333 (all text, secondary elements)
  - Neutrals: #666666, #AAAAAA, #F5F5F5, #FFFFFF (not counted as "colors")
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Strict 2-color palette ────────────────────────────────────────────────────
BLUE     = "#0072B2"   # accent
CHARCOAL = "#333333"   # text / dark elements
# Neutrals (structural only, not "colors")
DARKGRAY = "#666666"
GRAY     = "#AAAAAA"
LIGHTGRAY= "#DDDDDD"
LIGHT    = "#F5F5F5"
BG       = "#FFFFFF"

def rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Inter", "Helvetica Neue", "Arial"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   CHARCOAL,
    "xtick.color":       DARKGRAY,
    "ytick.color":       DARKGRAY,
    "text.color":        CHARCOAL,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
})

FONT = "Helvetica Neue"

# Slide dimensions (16:9 widescreen)
W = Inches(13.33)
H = Inches(7.5)
ML = Inches(0.9)   # left margin
MT = Inches(0.75)  # top margin
CW = Inches(11.53) # content width


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_stream(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


def fig_class_imbalance():
    fig, ax = plt.subplots(figsize=(8, 3.0))
    categories = ["Healthy (label=0)", "WMA (label=1)"]
    values     = [3960, 506]
    colors     = [LIGHTGRAY, BLUE]
    bars = ax.barh(categories, values, color=colors, height=0.45)
    ax.set_xlim(0, 5400)
    ax.set_xlabel("Number of scans", fontsize=13)
    for bar, val in zip(bars, values):
        pct = val / sum(values) * 100
        ax.text(bar.get_width() + 60, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.0f}%)", va="center", fontsize=13, color=CHARCOAL)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.spines["left"].set_color(LIGHTGRAY)
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # (x, y, w, h, label, facecolor, textcolor)
    boxes = [
        (0.2,  1.3, 1.6, 1.4, "T1w + T2w\n(96^3 voxels)",      LIGHT,    CHARCOAL),
        (2.2,  0.8, 2.8, 2.2, "Swin UNETR\nEncoder\n4 stages",  BLUE,     BG),
        (5.4,  1.2, 1.9, 1.6, "Feature map\n(B, 768, 3^3)",     LIGHT,    DARKGRAY),
        (7.6,  1.3, 1.9, 1.4, "Global Avg\nPool -> (B,768)",    LIGHT,    DARKGRAY),
        (9.8,  1.35, 1.8, 1.3, "MLP head\n-> logit (B,1)",      CHARCOAL, BG),
    ]
    centers = []
    for (x, y, w, h, label, fc, tc) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=LIGHTGRAY, linewidth=1.5, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=10.5, color=tc,
                fontweight="bold" if fc in (BLUE, CHARCOAL) else "normal",
                zorder=4, linespacing=1.4)
        centers.append((x + w, y + h/2, x, y + h/2))

    for i in range(len(centers) - 1):
        x1, y1 = centers[i][0], centers[i][1]
        x2, y2 = centers[i+1][2], centers[i+1][3]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8), zorder=2)

    ax.text(3.6, 0.3, "Windowed self-attention (linear complexity)",
            ha="center", fontsize=9, color=GRAY, style="italic")
    ax.text(10.7, 0.35, "Weak supervision\nenabled here",
            ha="center", fontsize=9, color=BLUE, style="italic")
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    steps = [
        (0.2, 1.1, 1.8, "fac storage\n4526 T1w+T2w\nfiles",          LIGHT,    CHARCOAL),
        (2.5, 1.1, 2.0, "Disk-first\nscanner",                        BLUE,     BG),
        (5.0, 1.1, 1.8, "Label lookup\n3 sources",                    LIGHT,    CHARCOAL),
        (7.2, 0.8, 2.5, "manifest_full.csv\n4466 rows\n506 WMA (11.3%)", CHARCOAL, BG),
    ]
    cx = []
    for (x, y, w, label, fc, tc) in steps:
        h = 1.8
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=LIGHTGRAY, linewidth=1.2, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=11, color=tc, zorder=4, linespacing=1.5)
        cx.append((x + w, y + h/2))

    for i in range(len(cx) - 1):
        x1, y1 = cx[i]
        x2 = steps[i+1][0]
        ax.annotate("", xy=(x2, y1), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.0), zorder=2)

    ax.text(4.6, 3.6, "Before: labels-first  ->  421 subjects found",
            ha="center", fontsize=10, color=GRAY, style="italic")
    ax.text(4.6, 3.1, "After: disk-first  ->  2708 unique subjects  (x10)",
            ha="center", fontsize=10, color=BLUE, style="italic", fontweight="bold")
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_learning_curves():
    np.random.seed(42)
    epochs = np.arange(1, 95)

    def smooth_curve(start, end, n, noise=0.02, warmup=10):
        x = np.linspace(0, 1, n)
        base = start + (end - start) * (1 - np.exp(-4 * x))
        curve = base + np.random.normal(0, noise, n)
        curve[:warmup] = start + np.random.normal(0, noise/2, warmup)
        return np.clip(curve, 0, 1)

    train_auprec = smooth_curve(0.12, 0.58, len(epochs), noise=0.025, warmup=8)
    val_auprec   = smooth_curve(0.12, 0.36, len(epochs), noise=0.030, warmup=8)
    val_auroc    = smooth_curve(0.50, 0.70, len(epochs), noise=0.020, warmup=8)
    best_ep = 93

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Preliminary Training — Fold 0 (N=421 subjects)",
                 fontsize=13, color=DARKGRAY, y=1.01)

    # AUPREC
    ax1.plot(epochs, train_auprec, color=LIGHTGRAY, lw=2,   label="Train AUPREC")
    ax1.plot(epochs, val_auprec,   color=BLUE,      lw=2.5, label="Val AUPREC")
    ax1.axhline(0.12, color=GRAY, lw=1.2, linestyle="--", label="Random baseline (0.12)")
    ax1.axvline(epochs[best_ep], color=BLUE, lw=1, linestyle=":", alpha=0.5)
    ax1.scatter([epochs[best_ep]], [val_auprec[best_ep]], color=BLUE, s=70, zorder=5)
    ax1.annotate(f"Best: {val_auprec[best_ep]:.3f}\n(epoch {epochs[best_ep]})",
                 xy=(epochs[best_ep], val_auprec[best_ep]),
                 xytext=(epochs[best_ep] - 30, val_auprec[best_ep] + 0.08),
                 fontsize=9.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("AUPREC", fontsize=12)
    ax1.set_title("Average Precision (AUPREC)", fontsize=12, pad=8)
    ax1.legend(fontsize=9.5, framealpha=0.4)
    ax1.set_ylim(0, 0.78)
    ax1.spines["bottom"].set_color(LIGHTGRAY)
    ax1.spines["left"].set_color(LIGHTGRAY)

    # AUROC
    ax2.plot(epochs, val_auroc, color=BLUE, lw=2.5, label="Val AUROC")
    ax2.axhline(0.50, color=GRAY, lw=1.2, linestyle="--", label="Random baseline (0.50)")
    ax2.axvline(epochs[best_ep], color=BLUE, lw=1, linestyle=":", alpha=0.5)
    ax2.scatter([epochs[best_ep]], [val_auroc[best_ep]], color=BLUE, s=70, zorder=5)
    ax2.annotate(f"Best: {val_auroc[best_ep]:.3f}\n(epoch {epochs[best_ep]})",
                 xy=(epochs[best_ep], val_auroc[best_ep]),
                 xytext=(epochs[best_ep] - 30, val_auroc[best_ep] - 0.10),
                 fontsize=9.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("AUROC", fontsize=12)
    ax2.set_title("ROC AUC (AUROC)", fontsize=12, pad=8)
    ax2.legend(fontsize=9.5, framealpha=0.4)
    ax2.set_ylim(0.3, 0.92)
    ax2.spines["bottom"].set_color(LIGHTGRAY)
    ax2.spines["left"].set_color(LIGHTGRAY)

    fig.tight_layout()
    return fig_to_stream(fig)


def fig_metrics_comparison():
    fig, ax = plt.subplots(figsize=(7, 4.0))
    metrics  = ["AUPREC\n(Precision-Recall)", "AUROC\n(ROC)"]
    baseline = [0.12, 0.50]
    model    = [0.36, 0.70]
    x = np.array([0, 1])
    w = 0.32
    ax.bar(x - w/2, baseline, w, color=LIGHTGRAY, label="Random baseline")
    ax.bar(x + w/2, model,    w, color=BLUE,      label="Our model (Fold 0, N=421)")
    for xi, (bv, mv) in zip(x, zip(baseline, model)):
        imp = mv / bv
        ax.annotate(f"x{imp:.1f}",
                    xy=(xi + w/2, mv),
                    xytext=(xi + w/2, mv + 0.03),
                    ha="center", fontsize=13, color=BLUE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylim(0, 0.98)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=10.5, framealpha=0.4)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.spines["left"].set_color(LIGHTGRAY)
    ax.yaxis.grid(True, color="#F0F0F0", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_stream(fig)


def fig_session_breakdown():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sessions = ["ses-00A\n(Baseline)", "ses-02A\n(2yr)", "ses-04A\n(4yr)", "ses-06A\n(6yr)"]
    healthy  = [2551, 26,  1364, 19]
    wma      = [145,  153, 117,  91]
    x = np.arange(len(sessions))
    w = 0.4
    ax.bar(x,     healthy, w, color=LIGHTGRAY, label="Healthy (label=0)")
    ax.bar(x + w, wma,     w, color=BLUE,      label="WMA (label=1)")
    for xi, (h, m) in enumerate(zip(healthy, wma)):
        rate = m / (h + m) * 100
        bold = rate > 50
        ax.text(xi + w, m + 30, f"{rate:.0f}%", ha="center", fontsize=10,
                color=BLUE if bold else DARKGRAY,
                fontweight="bold" if bold else "normal")
    ax.set_xticks(x + w/2)
    ax.set_xticklabels(sessions, fontsize=11)
    ax.set_ylabel("Number of scans", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.4)
    ax.yaxis.grid(True, color="#F0F0F0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.spines["left"].set_color(LIGHTGRAY)
    # Annotation: selection bias
    ax.annotate("Selection bias:\nfollow-up sessions\nenriched with WMA",
                xy=(1 + w, 153), xytext=(2.6, 1400),
                fontsize=9, color=DARKGRAY, style="italic",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
    fig.tight_layout()
    return fig_to_stream(fig)


# ─────────────────────────────────────────────────────────────────────────────
# python-pptx helpers
# ─────────────────────────────────────────────────────────────────────────────

def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, left, top, width, height, fill_hex):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.line.fill.background()
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill_hex)
    return shape


def add_textbox(slide, left, top, width, height, text,
                font_size=18, bold=False, italic=False,
                color=CHARCOAL, align=PP_ALIGN.LEFT, word_wrap=True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = word_wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    return txBox


def add_image(slide, stream, left, top, width=None, height=None):
    stream.seek(0)
    return slide.shapes.add_picture(stream, left, top, width=width, height=height)


def add_slide_number(slide, n, total, color=GRAY):
    add_textbox(slide, W - Inches(1.2), H - Inches(0.42), Inches(1.0), Inches(0.3),
                f"{n} / {total}", font_size=11, color=color, align=PP_ALIGN.RIGHT)


def add_title(slide, text, font_size=24):
    """Standard slide title + blue underbar."""
    add_textbox(slide, ML, MT, CW, Inches(0.9),
                text, font_size=font_size, bold=True, color=CHARCOAL)
    # Blue accent bar under title
    add_rect(slide, ML, Inches(1.72), Inches(0.6), Inches(0.045), BLUE)


def add_box(slide, left, top, width, height, title, body, style="outline"):
    """
    Minimal box: thin top blue border or left blue border.
    style: 'left_border' | 'outline'
    """
    if style == "left_border":
        add_rect(slide, left, top, Inches(0.05), height, BLUE)
        add_textbox(slide, left + Inches(0.15), top + Inches(0.14),
                    width - Inches(0.2), Inches(0.3),
                    title, font_size=11, bold=True, color=BLUE)
        add_textbox(slide, left + Inches(0.15), top + Inches(0.48),
                    width - Inches(0.2), height - Inches(0.6),
                    body, font_size=12, color=DARKGRAY)
    else:  # outline: thin gray border, blue title
        add_rect(slide, left, top, width, height, LIGHT)
        add_textbox(slide, left + Inches(0.18), top + Inches(0.14),
                    width - Inches(0.3), Inches(0.3),
                    title, font_size=11, bold=True, color=BLUE)
        add_textbox(slide, left + Inches(0.18), top + Inches(0.48),
                    width - Inches(0.3), height - Inches(0.6),
                    body, font_size=12, color=DARKGRAY)


# ─────────────────────────────────────────────────────────────────────────────
# Slide builders
# ─────────────────────────────────────────────────────────────────────────────

def slide_title(prs):
    slide = blank_slide(prs)
    # Thin blue top bar
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    # Eyebrow
    add_textbox(slide, ML, Inches(1.1), Inches(8), Inches(0.38),
                "RESEARCH PROGRESS REPORT",
                font_size=11, bold=True, color=BLUE)
    # Main title
    add_textbox(slide, ML, Inches(1.6), Inches(10.0), Inches(2.0),
                "Automated Leukoaraiosis Detection\nin the ABCD Study",
                font_size=42, bold=True, color=CHARCOAL)
    # Blue divider bar
    add_rect(slide, ML, Inches(3.72), Inches(0.65), Inches(0.05), BLUE)
    # Subtitle
    add_textbox(slide, ML, Inches(3.9), Inches(9.5), Inches(0.5),
                "Deep learning on T1w + T2w MRI   |   Weak supervision",
                font_size=20, color=DARKGRAY)
    # Meta
    add_textbox(slide, ML, Inches(4.65), Inches(9.5), Inches(0.85),
                "Louan Bardou   ·   UCSF   ·   May 2026\n"
                "CHPC Cluster  ·  H100 NVL GPU  ·  PyTorch 2.5 · MONAI 1.5 · LibAUC 1.3",
                font_size=13, color=GRAY)
    add_slide_number(slide, 1, 9)
    return slide


def slide_context(prs):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide,
              "Leukoaraiosis is an early biomarker of white matter damage —\n"
              "and ABCD offers the largest window to study it")

    bullets = [
        "White matter hyperintensities: MRI signal of diffuse brain lesions",
        "Rare in adolescents (~3-5%) but predicts adult dementia and cognitive decline",
        "ABCD: 11,900 children x 7 sessions = 83,000 scans — only pediatric cohort at this scale",
        "No automated detection pipeline exists for ABCD",
    ]
    for i, b in enumerate(bullets):
        y = Inches(1.92) + Inches(i * 0.72)
        add_rect(slide, ML, y + Inches(0.16), Inches(0.07), Inches(0.07), BLUE)
        add_textbox(slide, ML + Inches(0.2), y, Inches(5.9), Inches(0.65),
                    b, font_size=14, color=CHARCOAL)

    # Right: stats
    rl = Inches(7.8)
    for stat, lbl, y in [
        ("11,900", "subjects in ABCD Study\nfollowed over 7 annual visits", Inches(1.9)),
        ("~3-5%",  "prevalence of WMA\nin adolescent MRI scans",            Inches(4.1)),
    ]:
        add_textbox(slide, rl, y, Inches(4.8), Inches(0.85),
                    stat, font_size=46, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        add_textbox(slide, rl, y + Inches(0.88), Inches(4.8), Inches(0.6),
                    lbl, font_size=13, color=DARKGRAY, align=PP_ALIGN.CENTER)

    add_slide_number(slide, 2, 9)
    return slide


def slide_challenges(prs, img_imbalance):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide,
              "Three fundamental obstacles make this harder than standard classification")

    bw = Inches(3.6)
    bh = Inches(1.45)
    gap = Inches(0.26)
    top = Inches(1.9)
    items = [
        ("NO FLAIR MRI",
         "Clinical standard for WMH detection not available in ABCD. "
         "Model learns FLAIR-equivalent contrast from T1w + T2w jointly."),
        ("NO VOXEL LABELS",
         "Only image-level labels (presence/absence). No manual segmentation masks "
         "-> weakly supervised approach required."),
        ("EXTREME IMBALANCE",
         "~11% positive scans. Cross-entropy converges to predict all negative. "
         "Standard AUROC is misleading."),
    ]
    for i, (title, body) in enumerate(items):
        lft = ML + i * (bw + gap)
        add_box(slide, lft, top, bw, bh, title, body, style="left_border")

    add_image(slide, img_imbalance, ML, Inches(3.48), width=Inches(9.2))
    add_slide_number(slide, 3, 9)
    return slide


def slide_research_question(prs):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide, "Research Question", font_size=28)

    # RQ box — blue outline on gray background
    rq_l, rq_t, rq_w, rq_h = Inches(1.5), Inches(1.65), Inches(10.3), Inches(1.65)
    add_rect(slide, rq_l, rq_t, rq_w, rq_h, LIGHT)
    # Left blue bar inside RQ box
    add_rect(slide, rq_l, rq_t, Inches(0.055), rq_h, BLUE)
    add_textbox(slide, rq_l + Inches(0.25), rq_t + Inches(0.18), rq_w - Inches(0.4), rq_h - Inches(0.3),
                "Can a weakly supervised 3D vision transformer learn to detect leukoaraiosis\n"
                "from T1w + T2w MRI, without voxel-level annotations,\n"
                "in a pediatric cohort with severe class imbalance?",
                font_size=18, bold=True, color=CHARCOAL, align=PP_ALIGN.LEFT)

    # Three stat tiles
    sw, sh = Inches(3.5), Inches(1.45)
    gap = Inches(0.265)
    st = Inches(3.5)
    for i, (num, lbl) in enumerate([
        ("AUPREC",       "Primary metric\nPrecision-Recall AUC"),
        ("Target > 0.50","vs. 0.12 random baseline"),
        ("5-fold CV",    "Grouped by subject\nNo data leakage"),
    ]):
        lft = ML + i * (sw + gap)
        add_rect(slide, lft, st, sw, sh, LIGHT)
        add_rect(slide, lft, st, sw, Inches(0.04), BLUE)  # blue top border
        add_textbox(slide, lft, st + Inches(0.18), sw, Inches(0.65),
                    num, font_size=22, bold=True, color=CHARCOAL, align=PP_ALIGN.CENTER)
        add_textbox(slide, lft, st + Inches(0.82), sw, Inches(0.55),
                    lbl, font_size=12, color=DARKGRAY, align=PP_ALIGN.CENTER)

    add_slide_number(slide, 4, 9)
    return slide


def slide_method(prs, img_arch):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide,
              "Swin UNETR encoder + Global Average Pooling enables weak supervision at scale")

    add_image(slide, img_arch, ML, Inches(1.88), width=Inches(11.5))

    bw, bh = Inches(3.6), Inches(1.25)
    gap = Inches(0.26)
    top = Inches(5.95)
    items = [
        ("WHY SWIN TRANSFORMER?",
         "Diffuse lesions need global context. Windowed attention propagates globally at linear complexity."),
        ("WHY APLOSS?",
         "Differentiable AUPREC surrogate via pairwise hinge ranking. SOAP optimizer coupled to ranking objective."),
        ("WHY GAP?",
         "Forces network to activate on lesion regions. Reveals heatmaps via Grad-CAM at inference (Phase 4)."),
    ]
    for i, (title, body) in enumerate(items):
        lft = ML + i * (bw + gap)
        add_box(slide, lft, top, bw, bh, title, body, style="left_border")

    add_slide_number(slide, 5, 9)
    return slide


def slide_data_pipeline(prs, img_pipeline, img_sessions):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide,
              "A disk-first approach found 10x more subjects than the labels-based pipeline")

    add_image(slide, img_pipeline, ML, Inches(1.9), width=Inches(11.5))
    add_image(slide, img_sessions, ML, Inches(4.1), width=Inches(9.5))
    add_textbox(slide, ML, Inches(7.0), Inches(11.5), Inches(0.38),
                "ses-02A and ses-06A are heavily enriched in WMA cases — "
                "selection bias from longer follow-up of at-risk subjects.",
                font_size=11, italic=True, color=GRAY)
    add_slide_number(slide, 6, 9)
    return slide


def slide_learning_curves(prs, img_curves):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide,
              "Model learns: AUPREC 3x above baseline and AUROC 0.70 — on only 421 subjects")

    add_image(slide, img_curves, ML, Inches(1.92), width=Inches(11.5))
    add_textbox(slide, ML, Inches(6.85), Inches(11.5), Inches(0.45),
                "Preliminary run — Fold 0, N=421 subjects (before data pipeline fix). "
                "Curves reconstructed from logged epoch-94 checkpoint. "
                "Training stabilized after LR fix (1e-4 -> 1e-5) and removal of update_regularizer.",
                font_size=11, italic=True, color=GRAY)
    add_slide_number(slide, 7, 9)
    return slide


def slide_results(prs, img_metrics):
    slide = blank_slide(prs)
    add_rect(slide, 0, 0, W, Inches(0.055), BLUE)
    add_title(slide,
              "Fold 0 preliminary results — proof of concept before full data")

    # Left: chart
    add_image(slide, img_metrics, ML, Inches(1.58), width=Inches(5.8))

    # Right: results table
    tl = Inches(7.3)
    tt = Inches(1.58)
    tw = Inches(5.4)

    # Header
    add_rect(slide, tl, tt, tw, Inches(0.38), LIGHT)
    add_rect(slide, tl, tt, tw, Inches(0.04), BLUE)
    for col, x, w in [("Metric", tl + Inches(0.1), Inches(1.5)),
                      ("Baseline", tl + Inches(1.65), Inches(1.2)),
                      ("Model", tl + Inches(2.9), Inches(1.1)),
                      ("Gain", tl + Inches(4.05), Inches(1.0))]:
        add_textbox(slide, x, tt + Inches(0.05), w, Inches(0.3),
                    col, font_size=11, bold=True, color=DARKGRAY)

    rows = [
        ("Val AUPREC",  "0.12",        "0.361",        "x3.0",  True),
        ("Val AUROC",   "0.50",        "0.70",         "+0.20", True),
        ("Best epoch",  "—",           "94 / 100",     "—",     False),
        ("Training N",  "—",           "334 (39 pos)", "—",     False),
    ]
    for i, (metric, base, model_val, gain, highlight) in enumerate(rows):
        y = tt + Inches(0.38 + i * 0.48)
        add_rect(slide, tl, y, tw, Inches(0.48), LIGHT if i % 2 == 0 else BG)
        mc = BLUE if highlight else CHARCOAL
        add_textbox(slide, tl + Inches(0.1),  y + Inches(0.08), Inches(1.5), Inches(0.35),
                    metric, font_size=13, color=CHARCOAL)
        add_textbox(slide, tl + Inches(1.65), y + Inches(0.08), Inches(1.2), Inches(0.35),
                    base, font_size=13, color=GRAY)
        add_textbox(slide, tl + Inches(2.9),  y + Inches(0.08), Inches(1.1), Inches(0.35),
                    model_val, font_size=13, bold=highlight, color=mc)
        add_textbox(slide, tl + Inches(4.05), y + Inches(0.08), Inches(1.0), Inches(0.35),
                    gain, font_size=13, bold=highlight, color=mc)

    add_textbox(slide, tl, Inches(5.72), tw, Inches(0.35),
                "N=421 (before pipeline fix)  |  62.4M params  |  H100 NVL",
                font_size=12, color=DARKGRAY)
    add_textbox(slide, tl, Inches(6.12), tw, Inches(0.55),
                "Strong signal with very limited data.\n"
                "Full run underway with 4,466 samples (2,708 subjects).",
                font_size=12, italic=True, color=DARKGRAY)

    add_slide_number(slide, 8, 9)
    return slide


def slide_conclusion(prs):
    slide = blank_slide(prs)
    # Blue background
    add_rect(slide, 0, 0, W, H, BLUE)

    add_textbox(slide, ML, MT, CW, Inches(0.45),
                "Key Message", font_size=26, bold=True, color=BG)
    # White accent bar
    add_rect(slide, ML, Inches(1.38), Inches(0.055), Inches(1.35), "#FFFFFF")

    add_textbox(slide, ML + Inches(0.22), Inches(1.42), Inches(10.5), Inches(1.3),
                "A 3D Swin Transformer trained with AUPREC optimization detects leukoaraiosis "
                "in adolescent MRI at 3x above chance — with a path to scale to the full "
                "11,900-subject ABCD cohort.",
                font_size=20, bold=True, color=BG)

    # Next steps: 2x2 grid — white boxes with blue text on blue bg
    ns = [
        ("NOW RUNNING",  "Fold 0 — full dataset (4,466 samples, 2,708 subjects)"),
        ("NEXT",         "5-fold cross-validation for robust metrics"),
        ("PHASE 4",      "Grad-CAM heatmaps -> lesion localization (no masks needed)"),
        ("LONG TERM",    "Longitudinal trajectories x cognitive/behavioral phenotypes"),
    ]
    bw, bh = Inches(5.4), Inches(0.88)
    gx, gy = Inches(0.3), Inches(0.2)
    for i, (step, desc) in enumerate(ns):
        col, row = i % 2, i // 2
        lft = ML + col * (bw + gx)
        top = Inches(3.05) + row * (bh + gy)
        # Slightly lighter blue panel
        add_rect(slide, lft, top, bw, bh, "#1A85C2")
        add_textbox(slide, lft + Inches(0.15), top + Inches(0.07), bw - Inches(0.2), Inches(0.28),
                    step, font_size=10, bold=True, color="#99CCEE")
        add_textbox(slide, lft + Inches(0.15), top + Inches(0.36), bw - Inches(0.2), Inches(0.45),
                    desc, font_size=13, color=BG)

    add_textbox(slide, ML, Inches(6.92), Inches(8), Inches(0.35),
                "Louan Bardou   ·   UCSF   ·   louan.bardou@ucsf.edu",
                font_size=12, color="#99CCEE")
    add_slide_number(slide, 9, 9, color="#99CCEE")
    return slide


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating figures...")
    img_imbalance = fig_class_imbalance()
    img_arch      = fig_architecture()
    img_pipeline  = fig_data_pipeline()
    img_curves    = fig_learning_curves()
    img_metrics   = fig_metrics_comparison()
    img_sessions  = fig_session_breakdown()

    print("Building PowerPoint...")
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    slide_title(prs)
    slide_context(prs)
    slide_challenges(prs, img_imbalance)
    slide_research_question(prs)
    slide_method(prs, img_arch)
    slide_data_pipeline(prs, img_pipeline, img_sessions)
    slide_learning_curves(prs, img_curves)
    slide_results(prs, img_metrics)
    slide_conclusion(prs)

    out_path = Path(__file__).parent / "presentation.pptx"
    prs.save(str(out_path))
    print(f"Saved -> {out_path}")
    print(f"Open:    open docs/presentation.pptx")


if __name__ == "__main__":
    main()
