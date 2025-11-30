import os
from pypdf import PdfReader
from chapter_map import CHAPTERS, PDF_PATH

# Purpose:
#   Given a source PDF and a mapping of chapter names → (start_page, end_page),
#   this script extracts the text for each chapter and saves each chapter as
#   an individual UTF-8 text file under data/chapters/.

# Notes:
#   - Uses PyPDF's PdfReader to access PDF pages.
#   - The page numbers in CHAPTERS are assumed to be 0-indexed.
#   - Each chapter is extracted by iterating through its page range, inclusive
#     of start_page and exclusive of end_page (Python range convention).

def extract_chapters():
    """Extract each chapter from the input PDF and write it to its own .txt file."""
    
    # Load the PDF file (PDF_PATH comes from chapter_map.py)
    # "../" is used because this script runs inside /src or similar.
    reader = PdfReader(f"../{PDF_PATH}")
    
    # Create the output directory if it does not already exist
    # (keeps chapter text files organized)
    os.makedirs("data/chapters", exist_ok=True)
    
    # Loop through each (chapter_name, (start_page, end_page)) entry
    for chapter_name, (start_page, end_page) in CHAPTERS:
        print(f"Extracting {chapter_name} (pages {start_page}-{end_page})")
        
        chapter_text = ""
        
        # Iterate through the page range for this chapter
        # NOTE: range(start_page, end_page) includes start_page, excludes end_page
        for page_num in range(start_page, end_page):
            
            # Safety check: avoid indexing past the PDF page count
            if page_num < len(reader.pages):
                page = reader.pages[page_num]
                
                # Extract raw text from the page
                # .extract_text() returns the textual content detected on the page
                chapter_text += page.extract_text() + "\n\n"
        
        # Build final output path for this chapter
        filename = f"../data/chapters/{chapter_name}.txt"
        
        # Write the accumulated chapter text to a UTF-8 file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        
        print(f"Saved {chapter_name} to {filename}")


# Run extraction only if this script is executed directly (not imported)
if __name__ == "__main__":
    extract_chapters()
