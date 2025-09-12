# app/utils/zip_utils.py
from pathlib import Path
import io
import zipfile
import os

def zip_dir_and_get_bytes(folder: Path) -> bytes:
    """
    Create an in-memory ZIP of `folder` and return bytes.
    Keeps relative paths inside the archive.

    Args:
        folder: Path to directory to zip.

    Returns:
        bytes: ZIP file content.
    """
    folder = Path(folder)
    mem_zip = io.BytesIO()

    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                try:
                    arcname = str(file_path.relative_to(folder))
                except Exception:
                    arcname = file_path.name
                zf.write(file_path, arcname)
    mem_zip.seek(0)
    return mem_zip.read()