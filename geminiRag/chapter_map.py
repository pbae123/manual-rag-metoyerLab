# chapter_map.py

# Offset accounts for front matter pages (like title, index, contents)
# which are present in the PDF but not counted in the main document's page numbers.
OFFSET = 18

# CHAPTERS maps the chapter ID to the (start_page, end_page) for PyPDF.
# These page numbers are 0-indexed relative to the start of the PDF file.
CHAPTERS = [
    ("ch01", (1 + OFFSET, 37 + OFFSET)),
    ("ch02", (37 + OFFSET, 74 + OFFSET)),
    ("ch03", (74 + OFFSET, 123 + OFFSET)),
    ("ch04", (123 + OFFSET, 162 + OFFSET)),
    ("ch05", (162 + OFFSET, 217 + OFFSET)),
    ("ch06", (217 + OFFSET, 258 + OFFSET)),
    ("ch07", (258 + OFFSET, 299 + OFFSET)),
]

# PDF_PATH is the location of the source document, relative to the project root.
PDF_PATH = "data/The-Design-of-Everyday-Things-Revised-and-Expanded-Edition.pdf"