import io
import json
import random
import logging
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from fuzzywuzzy import fuzz
from textwrap import dedent
import re

import windows_mcp.uia as uia
from fastmcp.utilities.types import Image as McpImage

logger = logging.getLogger(__name__)


def clean_ocr_text(text: str) -> str:
    """
    clean for CJK languages:
    remove spaces between CJK characters that OCR might have inserted
    """

    cleaned = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", text)
    return cleaned


def _process_image_for_transfer(
    image: Image.Image, max_dimension: int = 1600, quality: int = 75
) -> bytes:
    """
    Compress the image and convert to JPEG format to reduce size
    """

    # Limit Image Transfer Size
    if image.width > max_dimension or image.height > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    # PNG -> JPEG
    image.convert("RGB").save(buffered, format="JPEG", quality=quality)
    img_bytes = buffered.getvalue()
    buffered.close()
    return img_bytes


async def _perform_ocr(
    screenshot: Image.Image, text_query: str
) -> List[Dict[str, Any]]:
    """uses Windows OCR APIs to find text in the given screenshot and return their bounding boxes.

    Args:
        screenshot (Image.Image): screenshot of the desktop to perform OCR on
        text_query (str): text to search for in the OCR results.

    Raises:
        RuntimeError: winrt not available
        RuntimeError: no OCR engine available

    Returns:
        List[Dict[str, Any]]: Matched text content with bounding boxes in the format {"text": str, "rect": (x, y, w, h)}
    """

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


async def locate_text_tool(
    desktop,
    text_query: str,
    use_vision: bool = False,
    region_hint: str = "all",
    occurrence_index: int | None = None,
):
    """Analyze the match results, calculate the center position, duplicate item deletion and match the output.

    Args:
        desktop (_type_): from services
        text_query (str): text to search for in the OCR results.
        use_vision (bool, optional): whether to return picture data. Defaults to False.
        region_hint (str, optional): use for partial pruning. Defaults to "all".
        occurrence_index (int | None, optional): the index of the specific occurrence to return. Defaults to None.

    Returns:
        A serialized list containing match information in dictionary form and labeled images.
        Each match dictionary includes the center point coordinates, bounding box dimensions, and a unique ID.
        If use_vision is True, an annotated image with bounding boxes and labels for each match is also returned.
    """

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

        is_fuzzy_match = fuzz.ratio(text_query_lower, text_lower) > 80

        if not (is_exact_in or is_fuzzy_match):
            continue

        if is_exact_in and len(text) > len(text_query) + 4:
            continue

        rx, ry, rw, rh = rect
        cx, cy = rx + rw / 2, ry + rh / 2

        # Spatial Pruning
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
        if use_vision:
            screenshot_bytes = _process_image_for_transfer(screenshot)
            return [
                json.dumps(
                    {
                        "status": "error",
                        "message": f"Text '{text_query}' not found. Please try region constraints or using a shorter query.",
                        "data": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                McpImage(data=screenshot_bytes, format="jpeg"),
            ]
        return [
            json.dumps(
                {
                    "status": "error",
                    "message": f"Text '{text_query}' not found. Please try region constraints or using a shorter query.",
                    "data": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        ]

    # Pre-calculate candidates
    candidates = []
    for i, match in enumerate(filtered_matches):
        candidates.append(
            {
                "center_point": match["center_point"],
                "bounds": match["bounds"],
                "id": i + 1,
                "text": match["text"],
                "rect_local": match["rect_local"],
            }
        )

    # Prepare Image if use_vision is True
    screenshot_bytes = None
    if use_vision:
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

        # Determine which candidates to draw
        targets_to_draw = candidates
        if (
            len(filtered_matches) == 1 or occurrence_index is not None
        ) and occurrence_index is not None:
            # If explicit index provided, only draw that one if valid
            idx = occurrence_index - 1
            if 0 <= idx < len(filtered_matches):
                targets_to_draw = [candidates[idx]]

        for cand in targets_to_draw:
            c_id = cand["id"]
            x, y, w, h = cand["rect_local"]

            # Offset applied for the padding in padded_screenshot
            rx, ry = x + padding, y + padding

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
                (label_x1 + 2, label_y1 + 2),
                label_text,
                fill=(255, 255, 255),
                font=font,
            )

        screenshot_bytes = _process_image_for_transfer(padded_screenshot)

    # Return data arrays without internal dict bloat
    output_candidates = [
        {
            "center_point": c["center_point"],
            "bounds": c["bounds"],
            "id": c["id"],
        }
        for c in candidates
    ]

    # Scenario A: Distinct Match / Explicit Index
    if len(filtered_matches) == 1 or occurrence_index is not None:
        idx = 0 if occurrence_index is None else occurrence_index - 1
        if idx < 0 or idx >= len(filtered_matches):
            response = [
                json.dumps(
                    {
                        "status": "error",
                        "message": f"occurrence_index {occurrence_index} out of bounds (found {len(filtered_matches)} matches).",
                        "data": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ]
            if use_vision and screenshot_bytes:
                response.append(McpImage(data=screenshot_bytes, format="jpeg"))
            return response

        match_data = output_candidates[idx]
        response = [
            json.dumps(
                {
                    "status": "clear",
                    "message": "found a clear match for the query.",
                    "data": [match_data],
                },
                ensure_ascii=False,
                indent=2,
            )
        ]
        if use_vision and screenshot_bytes:
            response.append(McpImage(data=screenshot_bytes, format="jpeg"))
        return response

    # Scenario B: Disambiguation / Set-of-Mark
    response_json = {
        "status": "ambiguous",
        "message": f"Multiple matching targets found (total: {len(output_candidates)}). Please refer to the numbered labels in the image to rerun this tool with a specified occurrence_index, or use coordinates listed in the data list below.",
        "data": output_candidates,
    }

    response = [json.dumps(response_json, ensure_ascii=False, indent=2)]
    if use_vision and screenshot_bytes:
        response.append(McpImage(data=screenshot_bytes, format="jpeg"))

    return response
