import os
from pypdf import PdfReader
from chapter_map import CHAPTERS, PDF_PATH

## extracts each chapter from the pdf and saves it to a separate text file
## need to make sure the pages are extracted correctly and the text is extracted correctly

def extract_chapters():
    """Extract each chapter from PDF to separate text files."""
    reader = PdfReader(f"../{PDF_PATH}")
    
    # Create chapters directory
    os.makedirs("data/chapters", exist_ok=True)
    
    for chapter_name, (start_page, end_page) in CHAPTERS:
        print(f"Extracting {chapter_name} (pages {start_page}-{end_page})")
        
        chapter_text = ""
        for page_num in range(start_page, end_page):
            if page_num < len(reader.pages):
                page = reader.pages[page_num]
                chapter_text += page.extract_text() + "\n\n"
        
        # Save to file
        filename = f"../data/chapters/{chapter_name}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        
        print(f"Saved {chapter_name} to {filename}")

if __name__ == "__main__":
    extract_chapters()
