import io
import json
import random
import logging
import colorsys
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from fuzzywuzzy import fuzz
from textwrap import dedent
import re

import windows_mcp.uia as uia
from fastmcp.utilities.types import Image as McpImage

logger = logging.getLogger(__name__)


# clean for CJK languages: remove spaces between CJK characters that OCR might have inserted
def clean_ocr_text(text: str) -> str:
    cleaned = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", text)
    return cleaned


async def _perform_ocr(
    screenshot: Image.Image, text_query: str
) -> List[Dict[str, Any]]:
    # Use winrt-Windows.Media.Ocr if available, else winsdk
    try:
        import winrt.windows.media.ocr as ocr
        import winrt.windows.graphics.imaging as imaging
        import winrt.windows.security.cryptography as crypto
        import winrt.windows.foundation.collections as collections
        import winrt.windows.globalization as globalization
    except ImportError:
        raise RuntimeError("winrt is missing.")
    # Extract the raw RGBA matrix
    rgba_image = screenshot.convert("RGBA")

    # Swap R and B Channels
    r, g, b, a = rgba_image.split()
    bgra_image = Image.merge("RGBA", (b, g, r, a))

    # image -> bytes
    raw_bytes = bgra_image.tobytes()

    # bytes -> IBuffer
    buffer = crypto.CryptographicBuffer.create_from_byte_array(raw_bytes)

    bmp = imaging.SoftwareBitmap(
        imaging.BitmapPixelFormat.BGRA8, bgra_image.width, bgra_image.height
    )
    bmp.copy_from_buffer(buffer)
    # ===================================================================
    languages = ocr.OcrEngine.available_recognizer_languages
    target_lang = None
    for lang in languages:
        if "en-GB" in lang.language_tag or "en-GB" in lang.language_tag:
            target_lang = lang
            break

    if target_lang:
        engine = ocr.OcrEngine.try_create_from_language(target_lang)
    else:
        engine = ocr.OcrEngine.try_create_from_user_profile_languages()

    if engine is None:
        raise RuntimeError(
            "No OCR engine available. Please ensure you have the appropriate OCR language packs installed in Windows settings."
        )
    result = await engine.recognize_async(bmp)

    matches = []
    for line in result.lines:
        raw_text = line.text
        line_text = clean_ocr_text(raw_text)
        # Use union over words for the boundaries of the whole line
        min_x = min(w.bounding_rect.x for w in line.words)
        min_y = min(w.bounding_rect.y for w in line.words)
        max_r = max(w.bounding_rect.x + w.bounding_rect.width for w in line.words)
        max_b = max(w.bounding_rect.y + w.bounding_rect.height for w in line.words)

        matches.append(
            {
                "text": line_text,
                "rect": (min_x, min_y, max_r - min_x, max_b - min_y),
            }
        )

        # Additional granularity: add words independently just in case
        for word in line.words:
            if word.text.strip():
                matches.append(
                    {
                        "text": word.text,
                        "rect": (
                            word.bounding_rect.x,
                            word.bounding_rect.y,
                            word.bounding_rect.width,
                            word.bounding_rect.height,
                        ),
                    }
                )

    return matches


def _check_color(
    image: Image.Image, rect: tuple[float, float, float, float], color_hint: str
) -> bool:
    x, y, w, h = rect
    # Sample the perimeter (bg) rather than the text (center) to avoid ClearType antialiasing issues
    samples = []
    step = 5
    for i in range(int(x), int(x + w), step):
        if 0 <= y - step < image.height and 0 <= i < image.width:
            samples.append(image.getpixel((i, int(y - step))))
        if 0 <= y + h + step < image.height and 0 <= i < image.width:
            samples.append(image.getpixel((i, int(y + h + step))))

    for j in range(int(y), int(y + h), step):
        if 0 <= x - step < image.width and 0 <= j < image.height:
            samples.append(image.getpixel((int(x - step), j)))
        if 0 <= x + w + step < image.width and 0 <= j < image.height:
            samples.append(image.getpixel((int(x + w + step), j)))

    if not samples:
        return True

    hsv_samples = [
        colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for r, g, b, *a in samples
    ]

    # Calculate mode / average components
    v_avg = sum(v for h, s, v in hsv_samples) / len(hsv_samples)

    if color_hint == "black":
        return v_avg < 0.3
    if color_hint == "white":
        return v_avg > 0.7

    hue_avg = sum(h for h, s, v in hsv_samples) / len(hsv_samples) * 360

    if color_hint == "red":
        return hue_avg < 20 or hue_avg > 340
    elif color_hint == "blue":
        return 180 <= hue_avg <= 260
    elif color_hint == "green":
        return 80 <= hue_avg <= 160
    elif color_hint == "yellow":
        return 40 <= hue_avg <= 70

    return True


async def locate_text_tool(
    desktop,
    text_query: str,
    region_hint: str = "all",
    color_hint: str = "any",
    occurrence_index: int | None = None,
):
    screenshot = desktop.get_screenshot()

    left_offset, top_offset, _, _ = uia.GetVirtualScreenRect()
    screen_w, screen_h = screenshot.width, screenshot.height

    matches = await _perform_ocr(screenshot, text_query)

    filtered_matches = []
    text_query_lower = text_query.lower()

    # Accuracy Scoring:
    def score_match(m):
        return abs(len(m["text"]) - len(text_query))

    # Closest Length Matches
    matches.sort(key=score_match)

    for match in matches:
        text, rect = match["text"], match["rect"]
        text_lower = text.lower()

        is_exact_in = text_query_lower in text_lower

        is_fuzzy_match = fuzz.ratio(text_query_lower, text_lower) > 75

        if not (is_exact_in or is_fuzzy_match):
            continue

        if is_exact_in and len(text) > len(text_query) + 4:
            continue

        rx, ry, rw, rh = rect
        cx, cy = rx + rw / 2, ry + rh / 2

        # 2. Spatial Pruning
        if region_hint == "top" and cy > screen_h / 2:
            continue
        if region_hint == "bottom" and cy < screen_h / 2:
            continue
        if region_hint == "left" and cx > screen_w / 2:
            continue
        if region_hint == "right" and cx < screen_w / 2:
            continue
        if region_hint == "center" and (
            cx < screen_w / 4
            or cx > screen_w * 3 / 4
            or cy < screen_h / 4
            or cy > screen_h * 3 / 4
        ):
            continue

        # 3. Visual Verification Pruning
        if color_hint != "any":
            if not _check_color(screenshot, rect, color_hint):
                continue

        bounds_dict = {
            "x": int(rx + left_offset),
            "y": int(ry + top_offset),
            "w": int(rw),
            "h": int(rh),
        }

        # Collision Detection-based Duplicate Removal
        is_duplicate_or_overlapped = False
        for f in filtered_matches:
            fb = f["bounds"]
            overlap = not (
                bounds_dict["x"] > fb["x"] + fb["w"]
                or bounds_dict["x"] + bounds_dict["w"] < fb["x"]
                or bounds_dict["y"] > fb["y"] + fb["h"]
                or bounds_dict["y"] + bounds_dict["h"] < fb["y"]
            )
            if overlap:
                is_duplicate_or_overlapped = True
                break

        if is_duplicate_or_overlapped:
            continue

        filtered_matches.append(
            {
                "text": text,
                "center_point": {"x": int(cx + left_offset), "y": int(cy + top_offset)},
                "bounds": bounds_dict,
                "rect_local": rect,
            }
        )

    if not filtered_matches:
        return [
            json.dumps(
                {
                    "status": "error",
                    "message": f"Text '{text_query}' not found. Please try reducing color/region constraints or using a shorter query.",
                }
            )
        ]

    # Scenario A: Distinct Match / Explicit Index
    # When explicit occurrence_index is provided OR there is only one match
    if len(filtered_matches) == 1 or occurrence_index is not None:
        idx = 0 if occurrence_index is None else occurrence_index - 1
        if idx < 0 or idx >= len(filtered_matches):
            return [
                json.dumps(
                    {
                        "status": "error",
                        "message": f"occurrence_index {occurrence_index} out of bounds (found {len(filtered_matches)} matches).",
                    }
                )
            ]

        match = filtered_matches[idx]
        return [
            json.dumps(
                {
                    "status": "success",
                    "data": {
                        "point": match["center_point"],
                        "bounds": match["bounds"],
                        "text_found": match["text"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        ]

    # Scenario B: Disambiguation / Set-of-Mark
    padding = 5
    width = int(screenshot.width + (1.5 * padding))
    height = int(screenshot.height + (1.5 * padding))
    padded_screenshot = Image.new("RGB", (width, height), color=(255, 255, 255))
    padded_screenshot.paste(screenshot, (padding, padding))

    draw = ImageDraw.Draw(padded_screenshot)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    candidates = []

    for i, match in enumerate(filtered_matches):
        c_id = i + 1
        x, y, w, h = match["rect_local"]

        # Offset applied for the padding in padded_screenshot
        rx, ry = x + padding, y + padding

        candidates.append(
            {
                "id": c_id,
                "center_point": match["center_point"],
                "label": f"[{match['text']}]",
            }
        )

        # Draw bounding rect
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        adjusted_box = (rx, ry, rx + w, ry + h)
        draw.rectangle(adjusted_box, outline=color, width=2)

        # Draw label ID tag
        label_text = str(c_id)
        label_width = draw.textlength(label_text, font=font)
        label_height = 14

        label_x1 = rx
        label_y1 = ry - label_height - 4
        label_x2 = label_x1 + label_width + 4
        label_y2 = ry

        draw.rectangle([(label_x1, label_y1), (label_x2, label_y2)], fill=color)
        draw.text(
            (label_x1 + 2, label_y1 + 2), label_text, fill=(255, 255, 255), font=font
        )

    # Limit Image Transfer Size
    max_dimension = 1600
    if (
        padded_screenshot.width > max_dimension
        or padded_screenshot.height > max_dimension
    ):
        padded_screenshot.thumbnail(
            (max_dimension, max_dimension), Image.Resampling.LANCZOS
        )

    buffered = io.BytesIO()
    # 2. PNG -> JPEG,quality = 75
    padded_screenshot.convert("RGB").save(buffered, format="JPEG", quality=75)
    screenshot_bytes = buffered.getvalue()
    buffered.close()

    response_json = {
        "status": "ambiguous",
        "message": f"Multiple matching targets found (total: {len(candidates)}). Please refer to the numbered labels in the image to rerun this tool with a specified `occurrence_index`, or use coordinates listed in the `candidates` list below.",
        "candidates": candidates,
    }

    return [
        json.dumps(response_json, ensure_ascii=False, indent=2),
        McpImage(data=screenshot_bytes, format="jpeg"),
    ]
