# rag_script.py
import os
import sys
from rag_utils import build_gemini_rag
from chapter_map import CHAPTERS # Import chapter mapping data

def main():
    # --- CONFIG: Define your base paths ---
    PDF_PATH = "data/The-Design-of-Everyday-Things-Revised-and-Expanded-Edition.pdf"
    CHAPTERS_DIR = "data/chapters"

    # --- CONSTANT: Hardcoded Model ---
    SELECTED_MODEL = "gemini-2.5-flash"

    while True:
        print("\n" + "="*40)
        print(" 🤖 GEMINI RAG CONTROLLER")
        print("="*40)
        print("1. Select Document")
        print("2. Quit Program")

        main_choice = input("\nSelect an option: ")

        if main_choice == "2" or main_choice.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            sys.exit()

        elif main_choice == "1":
            # --- DOCUMENT SELECTION MENU ---
            print("\n--- Available Documents ---")
            print("1. Full Book (PDF)")
            print("2. Chapter 1 (PDF)")
            print("3. Chapter 2 (PDF)")
            print("4. Chapter 3 (PDF)")
            print("5. Chapter 4 (PDF)")
            print("6. Chapter 5 (PDF)")
            print("7. Chapter 6 (PDF)")
            print("8. Chapter 7 (PDF)")
            print("9. Back to Main Menu")
            
            doc_choice = input("Select Document: ")
            
            file_to_load = ""
            collection_name = ""
            current_offset = 0 # Default offset for full book

            # --- SELECTION LOGIC ---
            if doc_choice == "1":
                file_to_load = PDF_PATH
                collection_name = "full-book-pdf"
                current_offset = 0
            
            # UPDATED: Now points to .pdf files AND fetches correct offset
            # CHAPTERS[i][1][0] gets the start_page from the tuple in chapter_map
            
            elif doc_choice == "2":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch01.pdf")
                collection_name = "ch01-pdf-data"
                current_offset = CHAPTERS[0][1][0] 
            
            elif doc_choice == "3":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch02.pdf")
                collection_name = "ch02-pdf-data"
                current_offset = CHAPTERS[1][1][0]

            elif doc_choice == "4":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch03.pdf")
                collection_name = "ch03-pdf-data"
                current_offset = CHAPTERS[2][1][0]

            elif doc_choice == "5":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch04.pdf")
                collection_name = "ch04-pdf-data"
                current_offset = CHAPTERS[3][1][0]

            elif doc_choice == "6":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch05.pdf")
                collection_name = "ch05-pdf-data"
                current_offset = CHAPTERS[4][1][0]

            elif doc_choice == "7":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch06.pdf")
                collection_name = "ch06-pdf-data"
                current_offset = CHAPTERS[5][1][0]

            elif doc_choice == "8":
                file_to_load = os.path.join(CHAPTERS_DIR, "ch07.pdf")
                collection_name = "ch07-pdf-data"
                current_offset = CHAPTERS[6][1][0]

            elif doc_choice == "9":
                continue 
            
            else:
                print("Invalid document choice.")
                continue

            # --- MODEL INITIALIZATION (Hardcoded) ---
            print(f"\n⚙️  Initializing RAG for '{collection_name}' using {SELECTED_MODEL}...")
            print(f"   (Applying Page Offset: {current_offset})")
            
            # Build the pipeline with the offset
            chain = build_gemini_rag(file_to_load, collection_name, SELECTED_MODEL, page_offset=current_offset)

            if not chain:
                print("❌ Setup failed. Returning to menu.")
                continue

            # --- CHAT LOOP ---
            print(f"\n✅ Chat Active! (Model: {SELECTED_MODEL})")
            
            while True:
                query = input(f"({SELECTED_MODEL} | type 'back' to menu) 🗣️  Question: ")
                
                if query.lower() == "back":
                    print("Returning to Main Menu...")
                    break
                
                if query.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    sys.exit()

                if not query.strip():
                    continue

                try:
                    print("⏳ Thinking...")
                    response = chain.invoke(query)
                    print(f"\n🤖 Answer:\n{response}")
                    print("-" * 50)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Tip: If you got a 404, this model version might not be available to your key.")
                    break

if __name__ == "__main__":
    main()