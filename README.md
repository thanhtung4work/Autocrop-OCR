# Autocrop OCR

A lightweight, end‑to‑end tool designed to detect documents in photos, apply perspective correction, enhance the image, and extract text using OCR. The goal is to provide a simple, fast, and reliable workflow for turning casual camera shots into clean, searchable documents.

## Installation

This project uses uv for package management.

1. Install dependencies
```bash
uv sync
```
2. After syncing, your environment will be ready to run the application.


## Directory Structure
```
data/
 ├── raw/         # Put your input images here
 ├── processed/   # Auto-generated processed images
 ├── result.pdf   # Searchable PDF output (default)
src/
 ├── #.py # utilities script
main.py      # Main application script
```

## Running the application

1. Place your input image(s) inside: `data/raw`
2. Run the main script:
```python
uv run python main.py
# or
python main.py
```
3. The generated searchable PDF will appear in: `data/final_document.pdf`

## Licence
MIT License — free to use and modify.