# extract_chapters.py
import os
from pypdf import PdfReader, PdfWriter
from chapter_map import CHAPTERS, PDF_PATH

def extract_chapters():
    """
    Slices the master PDF into individual chapter PDF files.
    Retains original formatting and page metadata.
    """
    
    try:
        print(f"Attempting to load PDF from: {PDF_PATH}")
        reader = PdfReader(PDF_PATH)
        print(f"PDF successfully loaded. Total pages: {len(reader.pages)}")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load the PDF file at {PDF_PATH}")
        print(f"Error: {e}")
        return
    
    output_dir = "data/chapters"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured at: {os.path.abspath(output_dir)}")
    
    # Loop through the map
    for chapter_name, (start_page, end_page) in CHAPTERS:
        print(f"\nProcessing {chapter_name} (Pages {start_page} to {end_page - 1})...")
        
        writer = PdfWriter()
        
        # Add the specific pages to the writer
        for i in range(start_page, end_page):
            if i < len(reader.pages):
                writer.add_page(reader.pages[i])
            else:
                print(f"Warning: Page {i} out of bounds.")
        
        # output filename is now .pdf
        filename = os.path.join(output_dir, f"{chapter_name}.pdf")
        
        with open(filename, "wb") as f:
            writer.write(f)
            
        print(f"✅ Saved {chapter_name}.pdf")

if __name__ == "__main__":
    extract_chapters()