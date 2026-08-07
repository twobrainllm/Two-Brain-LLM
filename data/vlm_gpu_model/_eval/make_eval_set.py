"""Generate a small VLM eval set with objective, pixel-derived ground truth.

Deliberately not a "describe this image" test. Every item has an answer that is
checkable by string match against a value computed from the pixels, so a
quantization comparison rests on something other than prose quality.

The existing `test_assets/test_image.png` is a weak probe -- its circle sits at
exactly (0.50W, 0.50H), so "centre" vs "lower-centre" is a judgement call and
both 4B and 8B were marked wrong on a technicality. Here, positions are pushed
to unambiguous corners and quadrants.

Pure stdlib: no PIL on this machine, so PNGs are written by hand via zlib.

    python make_eval_set.py     # writes images/ + ground_truth.json
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

W = H = 512
OUT = Path(__file__).parent
IMAGES = OUT / "images"

WHITE = (255, 255, 255)
RED = (215, 20, 20)
GREEN = (20, 170, 60)
BLUE = (30, 90, 220)
YELLOW = (240, 200, 30)
PURPLE = (140, 50, 190)
BLACK = (0, 0, 0)


def canvas() -> list[list[tuple[int, int, int]]]:
    return [[WHITE for _ in range(W)] for _ in range(H)]


def rect(px, x0, y0, x1, y1, color) -> None:
    for y in range(max(0, y0), min(H, y1)):
        for x in range(max(0, x0), min(W, x1)):
            px[y][x] = color


def disc(px, cx, cy, r, color) -> None:
    for y in range(max(0, cy - r), min(H, cy + r)):
        for x in range(max(0, cx - r), min(W, cx + r)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                px[y][x] = color


def tri_up(px, cx, cy, size, color) -> None:
    """Upward-pointing triangle, apex at (cx, cy-size)."""
    for i in range(size * 2):
        y = cy - size + i
        half = int(i * size / (size * 2) * 2) // 2 + i // 2
        rect(px, cx - half, y, cx + half, y + 1, color)


#: 7-segment layout: segments a,b,c,d,e,f,g per digit.
_SEG = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def digit(px, x, y, w, h, t, ch, color) -> None:
    """Draw a 7-segment digit -- gives OCR-ish testing with exact ground truth."""
    segs = _SEG[ch]
    if "a" in segs: rect(px, x, y, x + w, y + t, color)
    if "b" in segs: rect(px, x + w - t, y, x + w, y + h // 2, color)
    if "c" in segs: rect(px, x + w - t, y + h // 2, x + w, y + h, color)
    if "d" in segs: rect(px, x, y + h - t, x + w, y + h, color)
    if "e" in segs: rect(px, x, y + h // 2, x + t, y + h, color)
    if "f" in segs: rect(px, x, y, x + t, y + h // 2, color)
    if "g" in segs: rect(px, x, y + h // 2 - t // 2, x + w, y + h // 2 + t // 2, color)


def write_png(path: Path, px) -> None:
    raw = b"".join(
        b"\x00" + b"".join(bytes(px[y][x]) for x in range(W)) for y in range(H)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def build() -> list[dict]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    # 1. Position -- unambiguous opposite corners, not the centre.
    px = canvas()
    rect(px, 40, 40, 150, 150, BLUE)
    disc(px, 400, 400, 60, RED)
    write_png(IMAGES / "position.png", px)
    items.append({
        "image": "position.png",
        # The granularity has to be demanded explicitly. Asked openly, a model
        # may answer "square on the left, circle on the right" -- correct, but
        # coarser than the check, which would score it wrong for being vague
        # rather than for being mistaken.
        "question": (
            "Where is the blue square and where is the red circle? "
            "Answer using one of: top-left, top-right, bottom-left, bottom-right."
        ),
        "checks": [
            {"name": "square_top_left", "any_of": ["top left", "top-left", "upper left", "upper-left"]},
            {"name": "circle_bottom_right", "any_of": ["bottom right", "bottom-right", "lower right", "lower-right"]},
        ],
    })

    # 2. Counting -- 5 well-separated identical discs.
    px = canvas()
    for cx, cy in [(90, 90), (410, 90), (256, 256), (90, 410), (410, 410)]:
        disc(px, cx, cy, 45, GREEN)
    write_png(IMAGES / "count.png", px)
    items.append({
        "image": "count.png",
        "question": "How many green circles are in this image? Reply with just the number.",
        "checks": [{"name": "count_is_5", "any_of": ["5", "five"]}],
    })

    # 3. Colour + left-to-right order.
    px = canvas()
    rect(px, 30, 180, 170, 330, RED)
    rect(px, 186, 180, 326, 330, GREEN)
    rect(px, 342, 180, 482, 330, BLUE)
    write_png(IMAGES / "order.png", px)
    items.append({
        "image": "order.png",
        "question": "List the colours of the three bars from left to right.",
        "checks": [{"name": "order_rgb", "ordered": ["red", "green", "blue"]}],
    })

    # 4. OCR-ish -- a two-digit number.
    px = canvas()
    digit(px, 130, 160, 100, 190, 26, "4", BLACK)
    digit(px, 280, 160, 100, 190, 26, "7", BLACK)
    write_png(IMAGES / "digits.png", px)
    items.append({
        "image": "digits.png",
        "question": "What number is shown in this image? Reply with just the number.",
        "checks": [{"name": "number_is_47", "any_of": ["47", "forty-seven", "forty seven"]}],
    })

    # 5. Relative size -- same colour, so size is the only discriminator.
    px = canvas()
    disc(px, 140, 256, 100, PURPLE)
    disc(px, 390, 256, 40, PURPLE)
    write_png(IMAGES / "size.png", px)
    items.append({
        "image": "size.png",
        "question": "Which purple circle is larger, the one on the left or the one on the right?",
        "checks": [{"name": "left_is_larger", "any_of": ["left"], "none_of": ["right is larger", "right one is larger"]}],
    })

    # 6. Spatial relation between two different shapes.
    px = canvas()
    tri_up(px, 256, 150, 70, YELLOW)
    rect(px, 196, 320, 316, 440, PURPLE)
    write_png(IMAGES / "relation.png", px)
    items.append({
        "image": "relation.png",
        "question": "Is the yellow triangle above or below the purple square?",
        "checks": [{"name": "triangle_above", "any_of": ["above"], "none_of": ["below the purple", "triangle is below"]}],
    })

    (OUT / "ground_truth.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items


if __name__ == "__main__":
    built = build()
    print(f"wrote {len(built)} eval items to {IMAGES}")
    for it in built:
        print(f"  {it['image']:16s} {it['question'][:60]}")
