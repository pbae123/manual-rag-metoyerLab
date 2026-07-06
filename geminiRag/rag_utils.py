# rag_utils.py
import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


@dataclass(frozen=True)
class RagPipeline:
    chain: Runnable
    retriever: BaseRetriever
    embeddings: Embeddings

def clean_document_text(documents):
    """Cleaning logic for documents."""
    cleaned_documents = []
    for doc in documents:
        cleaned_content = doc.page_content
        # Remove known junk patterns
        JUNK_PATTERNS = [
            r"9780465050659-text\.indd",
            r"THE DESIGN OF EVERYDAY THINGS",
            r"The Design of Everyday Things",
            r"\d+\s+The Design of Everyday Things",
            r"8/19/13 5:22 PM"
        ]
        for pattern in JUNK_PATTERNS:
            cleaned_content = re.sub(pattern, " ", cleaned_content, flags=re.IGNORECASE).strip()
        
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        cleaned_content = cleaned_content.replace('. ', '.\n\n')
        
        doc.page_content = cleaned_content.strip()
        cleaned_documents.append(doc)
    return cleaned_documents

def build_gemini_rag(
    file_path,
    collection_name,
    model_version="gemini-1.5-flash",
    page_offset=0,
    search_type="mmr",
    k=8,
    fetch_k=50,
):
    """
    Builds the RAG pipeline with Citation Support.
    Args:
        file_path: a single path, or a list of paths, to .pdf/.txt source files.
        page_offset (int): The starting page index (0-based) of this file relative to the original book.
        search_type: retriever search type ("mmr" or "similarity").
        k: number of chunks to retrieve.
        fetch_k: candidate pool size for MMR (ignored for "similarity").
    """
    file_paths = file_path if isinstance(file_path, list) else [file_path]

    all_raw_docs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ Error: File not found: {path}")
            return None

        print(f"📖 Loading {path} into collection: '{collection_name}'...")

        # --- Use PyPDFLoader to capture Page Metadata ---
        if path.endswith(".pdf"):
            print("   Detected PDF format (scanning for page numbers)...")
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            print("   Detected Text format.")
            loader = TextLoader(path, encoding="utf-8")
        else:
            print("❌ Error: Unsupported file format.")
            return None

        all_raw_docs.extend(loader.load())

    cleaned_docs = clean_document_text(all_raw_docs)

    # Split text (Metadata is preserved during split)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(cleaned_docs)
    
    filtered_chunks = [c for c in chunks if len(c.page_content) > 50]

    # Embed & Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    vectorstore = Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        collection_name=collection_name, 
        persist_directory="./chroma_db"
    )

    if search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k}
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )
    
    # --- PROMPT: Combines Grounded Reasoning with Citations ---
    template = """
    You are an expert on 'The Design of Everyday Things'. 
    Your answers must be grounded in the provided context, but you should use 
    logical reasoning to connect the dots.
    
    Guidelines:
    1. Base your answer PRIMARILY on the provided context.
    2. You MUST cite the Page Number for every key fact you mention.
    3. If the answer is not explicitly written in one sentence, synthesize 
       information from multiple parts of the context to form an answer.
    4. If you quote a specific sentence, put it in quotes and cite the page.
    5. Do not make up new principles that are not in the book.
    
    Format example:
    "Explanation of the concept (Page 55). As explicitly stated: 'The quote from the book' (Page 56)."

    Context: 
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatGoogleGenerativeAI(
        model=model_version, 
        temperature=0, 
        google_api_key=GOOGLE_API_KEY
    )
    
    # --- Custom Formatter to Inject Metadata into the Prompt ---
    def format_docs_with_citations(docs):
        formatted_text = []
        
        # --- PAGE OFFSET SETTING ---
        # Fixed: 158 (Raw) - 15 (Offset) = 143 (Book Page)
        BOOK_START_OFFSET = 19

        for doc in docs:
            # Check if page metadata exists
            if 'page' in doc.metadata:
                # 1. Get the local page index in the current file (0-based)
                local_page_index = doc.metadata['page']
                
                # 2. Adjust by the Chapter Offset (to get 0-based index in Master PDF)
                master_page_index = local_page_index + page_offset

                # 3. Convert to 1-based "Physical" Page Number
                raw_page_number = master_page_index + 1
                
                # 4. Apply the Book Offset to match the printed page numbers
                final_page = raw_page_number - BOOK_START_OFFSET
                
                # 5. Handle "Negative" pages (Preface, TOC, Covers)
                if final_page < 1:
                    source_tag = "[Intro/Preface]"
                else:
                    source_tag = f"[Page {final_page}]"
            else:
                source_tag = "[Source: Text File]"
            
            # Combine content with the tag
            formatted_text.append(f"{doc.page_content}\nSOURCE: {source_tag}")
            
        return "\n\n".join(formatted_text)

    chain = (
        {"context": retriever | format_docs_with_citations, "question": RunnablePassthrough()}
        | prompt | model | StrOutputParser()
    )

    return RagPipeline(chain=chain, retriever=retriever, embeddings=embeddings)