"""Export faithful visual evidence for the 65-view multi-scale TTA pipeline.

This module deliberately reuses the production transformation helpers from
``benchmark_multiscale_tta.py``.  It does not run PARSeq inference; its job is
to make every image-space operation inspectable and to assemble a presentation
diagram from the exact exported views.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for import_path in (ROOT, HERE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from benchmark_multiscale_tta import (  # noqa: E402
    ViewSpec,
    apply_center_zoom,
    build_specs,
    unwrap_plate_lines,
    upscale_small_image,
)
from preprocessing import get_preprocessing_config, preprocess_plate_image  # noqa: E402


DEFAULT_SOURCE = ROOT / "wrong_images" / "59DB05813.png"
DEFAULT_OUTPUT = HERE / "multiview_pipeline_images"
MODEL_DISPLAY_SIZE = (256, 64)  # 2x display of the PARSeq 128x32 input.


COLORS = {
    "navy": "#153C73",
    "blue": "#1976D2",
    "cyan": "#EAF5FF",
    "green": "#2E7D32",
    "light_green": "#EAF6EA",
    "orange": "#E87500",
    "light_orange": "#FFF4E7",
    "ink": "#172033",
    "muted": "#5B6880",
    "line": "#A9BDD6",
    "paper": "#F7FAFE",
    "white": "#FFFFFF",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / max(image.width, 1), size[1] / max(image.height, 1))
    fitted_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    copy = image.resize(fitted_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _render_spec(image: Image.Image, spec: ViewSpec) -> tuple[Image.Image, Image.Image]:
    """Return geometry-stage and final model-input visualizations for one spec."""

    working = apply_center_zoom(image, spec.zoom)
    working = upscale_small_image(working, spec.upscale)
    if spec.unwrap_two_line:
        working = unwrap_plate_lines(working)
    geometry = working.copy()
    processed = preprocess_plate_image(working, get_preprocessing_config(spec.preprocessing))
    processed = processed.resize(MODEL_DISPLAY_SIZE, Image.Resampling.BICUBIC)
    return geometry, processed


def _contact_sheet(
    entries: Iterable[tuple[str, Image.Image]],
    columns: int,
    thumb_size: tuple[int, int] = MODEL_DISPLAY_SIZE,
    label_height: int = 36,
    gap: int = 16,
    margin: int = 24,
) -> Image.Image:
    entries = list(entries)
    rows = max(1, math.ceil(len(entries) / columns))
    cell_w = thumb_size[0]
    cell_h = thumb_size[1] + label_height
    width = margin * 2 + columns * cell_w + (columns - 1) * gap
    height = margin * 2 + rows * cell_h + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    label_font = _font(18, bold=True)
    for index, (label, image) in enumerate(entries):
        row, col = divmod(index, columns)
        x = margin + col * (cell_w + gap)
        y = margin + row * (cell_h + gap)
        draw.rounded_rectangle(
            (x - 2, y - 2, x + cell_w + 2, y + cell_h + 2),
            radius=8,
            fill=COLORS["white"],
            outline=COLORS["line"],
            width=2,
        )
        canvas.paste(_fit(image, thumb_size), (x, y))
        draw.text((x + cell_w / 2, y + thumb_size[1] + 7), label, font=label_font, fill=COLORS["ink"], anchor="ma")
    return canvas


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: str, max_width: int, line_gap: int = 6) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    x, y = xy
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = COLORS["blue"], width: int = 6) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 20
    wing = 0.55
    points = [
        end,
        (end[0] - length * math.cos(angle - wing), end[1] - length * math.sin(angle - wing)),
        (end[0] - length * math.cos(angle + wing), end[1] - length * math.sin(angle + wing)),
    ]
    draw.polygon(points, fill=color)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, accent: str, fill: str = COLORS["white"]) -> None:
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=accent, width=4)
    x0, y0, x1, _ = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 58), radius=20, fill=accent)
    draw.rectangle((x0, y0 + 36, x1, y0 + 58), fill=accent)
    draw.text((x0 + 22, y0 + 29), title, font=_font(25, bold=True), fill=COLORS["white"], anchor="lm")


def _small_image_grid(images: list[Image.Image], size: tuple[int, int], columns: int, gap: int = 8) -> Image.Image:
    rows = math.ceil(len(images) / columns)
    cell_w = (size[0] - gap * (columns - 1)) // columns
    cell_h = (size[1] - gap * (rows - 1)) // rows
    grid = Image.new("RGB", size, COLORS["paper"])
    for i, image in enumerate(images):
        row, col = divmod(i, columns)
        grid.paste(_fit(image, (cell_w, cell_h)), (col * (cell_w + gap), row * (cell_h + gap)))
    return grid


def _build_pipeline_figure(
    source: Image.Image,
    baseline: Image.Image,
    full_examples: list[Image.Image],
    unwrap_geometry: Image.Image,
    unwrap_examples: list[Image.Image],
    all_views: list[Image.Image],
) -> Image.Image:
    width, height = 3600, 1260
    canvas = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 48), "MULTI-VIEW PREPROCESSING & CONSENSUS", font=_font(52, bold=True), fill=COLORS["navy"], anchor="ma")
    draw.text((width // 2, 110), "65-VIEW MULTI-SCALE TTA — faithful to benchmark_multiscale_tta.py", font=_font(25), fill=COLORS["muted"], anchor="ma")

    source_box = (45, 190, 410, 1050)
    baseline_box = (520, 190, 1120, 430)
    full_box = (520, 485, 1120, 780)
    unwrap_box = (520, 835, 1120, 1185)
    total_box = (1250, 260, 2150, 1110)
    model_box = (2300, 400, 2650, 910)
    pred_box = (2800, 340, 3190, 970)
    consensus_box = (3320, 450, 3555, 850)

    _panel(draw, source_box, "INPUT PLATE CROP", COLORS["navy"])
    canvas.paste(_fit(source, (315, 420), COLORS["paper"]), (70, 270))
    draw.text((227, 735), f"{source.width} × {source.height}px", font=_font(24, bold=True), fill=COLORS["ink"], anchor="ma")
    aspect = source.width / max(source.height, 1)
    draw.text((227, 780), f"aspect = {aspect:.2f}", font=_font(23), fill=COLORS["muted"], anchor="ma")
    route = "two-line candidate" if aspect < 1.9 else "single-line geometry"
    draw.rounded_rectangle((88, 840, 366, 905), radius=14, fill=COLORS["cyan"], outline=COLORS["blue"], width=2)
    draw.text((227, 873), route, font=_font(23, bold=True), fill=COLORS["blue"], anchor="mm")
    _draw_wrapped(draw, (80, 945), "The baseline is always emitted. Unwrap is conditional inside 24 view specs — it is not a second baseline.", _font(19), COLORS["muted"], 300)

    _panel(draw, baseline_box, "1 BASELINE VIEW", COLORS["navy"], COLORS["white"])
    canvas.paste(_fit(baseline, (330, 83)), (550, 290))
    draw.text((920, 320), "zoom 1.00", font=_font(21, bold=True), fill=COLORS["ink"])
    draw.text((920, 353), "upscale 1×", font=_font(21), fill=COLORS["muted"])
    draw.text((920, 386), "train_baseline", font=_font(21), fill=COLORS["muted"])

    _panel(draw, full_box, "40 FULL VIEWS", COLORS["blue"], COLORS["white"])
    full_grid = _small_image_grid(full_examples, (535, 145), columns=5, gap=8)
    canvas.paste(full_grid, (552, 555))
    draw.text((820, 730), "2 upscales × 5 zooms × 4 preprocessors = 40", font=_font(21, bold=True), fill=COLORS["blue"], anchor="ma")

    _panel(draw, unwrap_box, "24 CONDITIONAL UNWRAP VIEWS", COLORS["orange"], COLORS["white"])
    canvas.paste(_fit(unwrap_geometry, (300, 108)), (550, 910))
    unwrap_grid = _small_image_grid(unwrap_examples, (225, 108), columns=3, gap=7)
    canvas.paste(unwrap_grid, (875, 910))
    draw.text((820, 1055), "split at low-stroke valley → top line | bottom line", font=_font(20, bold=True), fill=COLORS["orange"], anchor="ma")
    draw.text((820, 1100), "2 upscales × 3 zooms × 4 preprocessors = 24", font=_font(21, bold=True), fill=COLORS["orange"], anchor="ma")
    draw.text((820, 1142), "If aspect ≥ 1.9, unwrap returns the input unchanged", font=_font(18), fill=COLORS["muted"], anchor="ma")

    _panel(draw, total_box, "TOTAL = 1 + 40 + 24 = 65 MODEL INPUTS", COLORS["green"], COLORS["white"])
    total_grid = _small_image_grid(all_views, (820, 665), columns=8, gap=7)
    canvas.paste(total_grid, (1290, 355))
    draw.text((1700, 1060), "Every tile above is loaded from the notebook export folder", font=_font(21, bold=True), fill=COLORS["green"], anchor="ma")

    _panel(draw, model_box, "PARSeq", COLORS["navy"], COLORS["white"])
    draw.rounded_rectangle((2385, 515, 2565, 730), radius=28, fill=COLORS["cyan"], outline=COLORS["blue"], width=5)
    draw.arc((2430, 545, 2520, 635), 180, 360, fill=COLORS["navy"], width=13)
    draw.rounded_rectangle((2420, 600, 2530, 690), radius=13, fill=COLORS["navy"])
    draw.ellipse((2464, 626, 2486, 648), fill=COLORS["white"])
    draw.rectangle((2471, 642, 2479, 665), fill=COLORS["white"])
    draw.text((2475, 785), "fixed checkpoint", font=_font(24, bold=True), fill=COLORS["ink"], anchor="ma")
    draw.text((2475, 825), "65 forward passes", font=_font(21), fill=COLORS["muted"], anchor="ma")

    _panel(draw, pred_box, "PREDICTION + CONFIDENCE", COLORS["blue"], COLORS["white"])
    y = 445
    prediction_colors = [COLORS["navy"], COLORS["blue"], COLORS["green"], COLORS["orange"]]
    for i in range(9):
        draw.rounded_rectangle((2840, y, 3150, y + 40), radius=8, fill="#F1F6FC")
        draw.text((2860, y + 20), f"view {i + 1:02d}", font=_font(18, bold=True), fill=COLORS["muted"], anchor="lm")
        draw.text((2992, y + 20), "59DB05813", font=_font(18, bold=True), fill=prediction_colors[i % 4], anchor="lm")
        y += 51
    draw.text((2995, 920), "...  65 outputs", font=_font(24, bold=True), fill=COLORS["muted"], anchor="ma")

    _panel(draw, consensus_box, "CONSENSUS", COLORS["green"], COLORS["white"])
    draw.ellipse((3370, 560, 3505, 695), fill=COLORS["light_green"], outline=COLORS["green"], width=7)
    draw.line((3402, 628, 3438, 660, 3480, 592), fill=COLORS["green"], width=12, joint="curve")
    draw.text((3438, 755), "59DB05813", font=_font(22, bold=True), fill=COLORS["green"], anchor="ma")
    draw.text((3438, 795), "vote + confidence", font=_font(17), fill=COLORS["muted"], anchor="ma")

    _arrow(draw, (410, 315), (520, 315), COLORS["navy"])
    _arrow(draw, (410, 610), (520, 610), COLORS["blue"])
    _arrow(draw, (410, 995), (520, 995), COLORS["orange"])
    draw.line((458, 315, 458, 995), fill=COLORS["line"], width=5)
    _arrow(draw, (1120, 315), (1250, 420), COLORS["navy"])
    _arrow(draw, (1120, 635), (1250, 650), COLORS["blue"])
    _arrow(draw, (1120, 995), (1250, 885), COLORS["orange"])
    _arrow(draw, (2150, 685), (2300, 655), COLORS["green"])
    _arrow(draw, (2650, 655), (2800, 655), COLORS["blue"])
    _arrow(draw, (3190, 655), (3320, 655), COLORS["green"])
    return canvas


def export_visualization(
    source_path: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
    demo_size: tuple[int, int] | None = (96, 70),
) -> dict:
    """Export all 65 views, evidence sheets, manifest, and the final diagram."""

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source plate not found: {source_path}")
    full_dir = output_dir / "views" / "full"
    unwrap_dir = output_dir / "views" / "unwrap"
    geometry_dir = output_dir / "geometry"
    for directory in (output_dir, full_dir, unwrap_dir, geometry_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as opened:
        source_original = opened.convert("RGB")
    source = source_original.resize(demo_size, Image.Resampling.LANCZOS) if demo_size else source_original
    source.save(output_dir / "00_source_plate.png")
    source_original.save(output_dir / "00_source_plate_original_resolution.png")

    specs = build_specs()
    if len(specs) != 65:
        raise AssertionError(f"Expected 65 view specs, got {len(specs)}")
    full_specs = [spec for spec in specs if spec.name.startswith("full_")]
    unwrap_specs = [spec for spec in specs if spec.name.startswith("unwrap_")]
    if (len(full_specs), len(unwrap_specs)) != (40, 24):
        raise AssertionError(f"Expected 40 full and 24 unwrap specs, got {len(full_specs)} and {len(unwrap_specs)}")

    rows: list[dict] = []
    rendered: dict[str, Image.Image] = {}
    geometries: dict[str, Image.Image] = {}
    for index, spec in enumerate(specs):
        geometry, processed = _render_spec(source, spec)
        rendered[spec.name] = processed
        geometries[spec.name] = geometry
        if spec.name == "baseline":
            relative_path = Path("01_baseline.png")
        elif spec.unwrap_two_line:
            relative_path = Path("views") / "unwrap" / f"{spec.name}.png"
        else:
            relative_path = Path("views") / "full" / f"{spec.name}.png"
        processed.save(output_dir / relative_path)
        rows.append(
            {
                "index": index,
                "name": spec.name,
                "branch": "baseline" if spec.name == "baseline" else "unwrap" if spec.unwrap_two_line else "full",
                "zoom": f"{spec.zoom:.2f}",
                "upscale": f"{spec.upscale:.0f}",
                "preprocessing": spec.preprocessing,
                "unwrap_two_line": spec.unwrap_two_line,
                "output": relative_path.as_posix(),
            }
        )

    with (output_dir / "view_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "view_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    # Geometry evidence: the source and a representative unwrapped result.
    representative_unwrap = unwrap_specs[4]  # z1.00/up2/train_baseline
    unwrap_geometry = geometries[representative_unwrap.name]
    unwrap_geometry.save(geometry_dir / "unwrap_top_then_bottom.png")
    geometry_sheet = _contact_sheet(
        [("two-line input", source), ("top | bottom", unwrap_geometry)],
        columns=2,
        thumb_size=(360, 150),
        label_height=42,
    )
    geometry_sheet.save(output_dir / "02_unwrap_geometry.png")

    full_sheet_entries = [(f"{s.zoom:.2f} / {s.upscale:.0f}x / {s.preprocessing}", rendered[s.name]) for s in full_specs]
    unwrap_sheet_entries = [(f"{s.zoom:.2f} / {s.upscale:.0f}x / {s.preprocessing}", rendered[s.name]) for s in unwrap_specs]
    _contact_sheet(full_sheet_entries, columns=4).save(output_dir / "03_full_40_views.png")
    _contact_sheet(unwrap_sheet_entries, columns=4).save(output_dir / "04_unwrap_24_views.png")

    preprocessor_specs = [s for s in full_specs if s.upscale == 2.0 and abs(s.zoom - 1.0) < 1e-6]
    _contact_sheet([(s.preprocessing, rendered[s.name]) for s in preprocessor_specs], columns=2).save(output_dir / "05_four_preprocessors.png")
    all_sheet = _contact_sheet([(str(i + 1), rendered[s.name]) for i, s in enumerate(specs)], columns=5, label_height=30)
    all_sheet.save(output_dir / "06_all_65_views.png")

    full_examples = [rendered[s.name] for s in full_specs[::4]][:10]
    unwrap_examples = [rendered[s.name] for s in unwrap_specs[::4]][:6]
    final = _build_pipeline_figure(
        source,
        rendered["baseline"],
        full_examples,
        unwrap_geometry,
        unwrap_examples,
        [rendered[s.name] for s in specs],
    )
    final_path = output_dir / "multiview_65_pipeline.png"
    final.save(final_path)

    summary = {
        "source": str(source_path),
        "visualization_input_size": list(source.size),
        "source_was_downsampled_for_upscale_demo": demo_size is not None,
        "aspect_ratio": source.width / max(source.height, 1),
        "unwrap_is_active": source.width / max(source.height, 1) < 1.9,
        "baseline_views": 1,
        "full_views": len(full_specs),
        "unwrap_views": len(unwrap_specs),
        "total_views": len(specs),
        "output_dir": str(output_dir),
        "final_pipeline": str(final_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _parse_size(value: str) -> tuple[int, int] | None:
    if value.lower() in {"none", "original"}:
        return None
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--demo-size",
        type=_parse_size,
        default=(96, 70),
        help="Downsample WxH so the low-resolution 2x/3x path is visible; use 'original' to disable.",
    )
    args = parser.parse_args()
    summary = export_visualization(args.source, args.output_dir, args.demo_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
