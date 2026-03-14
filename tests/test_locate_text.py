import pytest
import json
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

from windows_mcp.desktop.locate_text import locate_text_tool
import windows_mcp.uia as uia


@pytest.fixture
def mock_desktop():
    desktop = MagicMock()

    # Create a dummy image for testing OCR (a simple white image)
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw some "fake text" (just for visuals, OCR will be mocked)
    draw.text((100, 100), "Click Here", fill=(0, 0, 0))
    draw.text((100, 400), "Click Here", fill=(0, 0, 255))  # Blue text

    desktop.get_screenshot.return_value = img
    return desktop


@pytest.fixture
def mock_uia_screen_rect():
    with patch("windows_mcp.desktop.locate_text.uia.GetVirtualScreenRect") as mock_rect:
        mock_rect.return_value = (0, 0, 1920, 1080)
        yield mock_rect


@pytest.mark.asyncio
async def test_locate_text_no_matches(mock_desktop, mock_uia_screen_rect):
    with patch("windows_mcp.desktop.locate_text._perform_ocr", return_value=[]):
        result = await locate_text_tool(mock_desktop, text_query="MissingText")
        assert len(result) == 1
        data = json.loads(result[0])
        assert data["status"] == "error"
        assert "not found" in data["message"]


@pytest.mark.asyncio
async def test_locate_text_single_match(mock_desktop, mock_uia_screen_rect):
    mock_ocr_result = [{"text": "Submit", "rect": (50, 50, 100, 40)}]
    with patch(
        "windows_mcp.desktop.locate_text._perform_ocr", return_value=mock_ocr_result
    ):
        # We also mock check_color just in case
        with patch("windows_mcp.desktop.locate_text._check_color", return_value=True):
            result = await locate_text_tool(mock_desktop, text_query="Submit")
            assert len(result) == 1
            data = json.loads(result[0])
            assert data["status"] == "success"

            assert data["data"]["point"]["x"] == 100
            assert data["data"]["point"]["y"] == 70
            assert data["data"]["bounds"]["w"] == 100


@pytest.mark.asyncio
async def test_locate_text_ambiguous(mock_desktop, mock_uia_screen_rect):
    mock_ocr_result = [
        {"text": "Button", "rect": (10, 10, 50, 20)},
        {"text": "Button", "rect": (10, 100, 50, 20)},
        {"text": "ButtonX", "rect": (10, 200, 50, 20)},
    ]
    with patch(
        "windows_mcp.desktop.locate_text._perform_ocr", return_value=mock_ocr_result
    ):
        result = await locate_text_tool(mock_desktop, text_query="Button")

        # Should return JSON and an Image
        assert len(result) == 2

        data = json.loads(result[0])
        assert data["status"] == "ambiguous"
        assert len(data["candidates"]) == 3
        assert data["candidates"][0]["id"] == 1
        assert data["candidates"][1]["id"] == 2
        assert data["candidates"][2]["id"] == 3

        # Second item should be the image
        assert hasattr(result[1], "data")
        assert getattr(result[1], "_format", None) == "jpeg"


@pytest.mark.asyncio
async def test_locate_text_ambiguous_with_index(mock_desktop, mock_uia_screen_rect):
    mock_ocr_result = [
        {"text": "Apply", "rect": (0, 0, 10, 10)},
        {"text": "Apply", "rect": (100, 100, 10, 10)},
    ]
    with patch(
        "windows_mcp.desktop.locate_text._perform_ocr", return_value=mock_ocr_result
    ):
        # When user passes occurrence_index=2, we bypass the ambiguous image branch
        result = await locate_text_tool(
            mock_desktop, text_query="Apply", occurrence_index=2
        )
        assert len(result) == 1

        data = json.loads(result[0])
        assert data["status"] == "success"
        # The 2nd item has bounds x=100
        assert data["data"]["bounds"]["x"] == 100


@pytest.mark.asyncio
async def test_locate_text_spatial_pruning(mock_desktop, mock_uia_screen_rect):
    # screen in mock is 800x600
    mock_ocr_result = [
        {"text": "TopNav", "rect": (10, 10, 50, 20)},  # Cy = 20
        {"text": "TopNav", "rect": (10, 500, 50, 20)},  # Cy = 510 (bottom)
    ]
    with patch(
        "windows_mcp.desktop.locate_text._perform_ocr", return_value=mock_ocr_result
    ):
        # We request only top half
        result = await locate_text_tool(
            mock_desktop, text_query="TopNav", region_hint="top"
        )

        # Since the second one is pruned, we are left with exactly 1 match -> Success!
        assert len(result) == 1
        data = json.loads(result[0])
        assert data["status"] == "success"
        assert data["data"]["point"]["y"] == 20


@pytest.mark.asyncio
async def test_locate_text_center_ambiguous_from_test_png_and_save_artifacts():
    test_png = (
        Path(__file__).parent.parent / "assets" / "screenshots" / "screenshot_1.png"
    )
    assert test_png.exists(), "screenshot_1.png not found"
    screenshot = Image.open(test_png).convert("RGB")
    width, height = screenshot.size

    desktop = MagicMock()
    desktop.get_screenshot.return_value = screenshot

    # Two matches in center region + one outside center (should be pruned)
    mock_ocr_result = [
        {
            "text": "Windows-MCP",
            "rect": (width // 2 - 220, height // 2 - 30, 170, 40),
        },
        {
            "text": "Windos-MCP",
            "rect": (width // 2 + 30, height // 2 + 10, 180, 40),
        },
        {
            "text": "Windows-MCP",
            "rect": (20, 20, 150, 30),
        },
    ]

    with patch(
        "windows_mcp.desktop.locate_text.uia.GetVirtualScreenRect",
        return_value=(0, 0, width, height),
    ):
        with patch(
            "windows_mcp.desktop.locate_text._perform_ocr", return_value=mock_ocr_result
        ):
            result = await locate_text_tool(
                desktop,
                text_query="Windows-MCP",
                region_hint="center",
            )

    assert len(result) == 2

    data = json.loads(result[0])
    assert data["status"] == "ambiguous"
    assert len(data["candidates"]) == 2

    candidate_ids = [c["id"] for c in data["candidates"]]
    assert candidate_ids == [1, 2], f"Expected IDs [1, 2], but got {candidate_ids}"

    image_bytes = getattr(result[1], "data", None)
    assert image_bytes

    base64_image = base64.b64encode(image_bytes).decode("ascii")
    preview_payload = {
        "status": data["status"],
        "message": data["message"],
        "candidates": data["candidates"],
        "image_attachment": f"data:image/png;base64,{base64_image}",
    }


@pytest.mark.asyncio
async def test_real_ocr_integration_and_save_artifacts():
    """
    mark for screenshot_1.png。
    """
    test_png = (
        Path(__file__).parent.parent / "assets" / "screenshots" / "screenshot_1.png"
    )

    screenshot = Image.open(test_png).convert("RGB")
    width, height = screenshot.size

    desktop = MagicMock()
    desktop.get_screenshot.return_value = screenshot

    with patch(
        "windows_mcp.desktop.locate_text.uia.GetVirtualScreenRect",
        return_value=(0, 0, width, height),
    ):
        result = await locate_text_tool(
            desktop,
            text_query="Claude",
            region_hint="all",
            color_hint="any",
        )

    assert len(result) > 0, "OCR finds no matches"
    data = json.loads(result[0])

    if data["status"] == "ambiguous":
        assert len(result) == 2, "ambiguous result should include image"
        image_bytes = getattr(result[1], "data", None)
        assert image_bytes

    elif data["status"] == "success":
        pytest.fail(
            f"expected ambiguous due to multiple 'Tools' matches, but got success"
        )
    else:
        pytest.fail(f"can't find '{data.get('message')}'")
