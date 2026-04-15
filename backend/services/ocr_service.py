from __future__ import annotations

import os

from PIL import Image, ImageFilter, ImageOps
import pytesseract


OCR_LANGS = os.getenv("OCR_LANGS", "ukr+rus+eng")
OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "60"))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    width, height = gray.size
    if max(width, height) < 1800:
        gray = gray.resize((width * 2, height * 2), Image.LANCZOS)

    gray = gray.filter(ImageFilter.SHARPEN)

    bw = gray.point(lambda x: 0 if x < 160 else 255, mode="1")
    return bw.convert("RGB")


def image_to_text(image: Image.Image, lang: str | None = None) -> str:
    processed = preprocess_image(image)

    text = pytesseract.image_to_string(
        processed,
        lang=lang or OCR_LANGS,
        config="--oem 1 --psm 6",
        timeout=OCR_TIMEOUT,
    )

    return text.strip()