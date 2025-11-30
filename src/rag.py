import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from rag_utils import build_rag


def main():
    #allows you to access environment variables
    load_dotenv()

    CHAPTERS_DIR = "data/chapters"

    #get all chapter filenames (alphabetically sorted)
    chapter_files = sorted(
        [f for f in os.listdir(CHAPTERS_DIR) if f.endswith(".txt")]
    )

    if not chapter_files:
        print("No chapter files found in data/chapters/")
        return

    print("\n=== RAG System Menu ===")
    print("1. Full-book RAG (all chapters)")
    print("2. Chapter 2 only")
    print("3. Chapter 4 only\n")

    choice = input("Choose a mode: ")

    #build RAG based on user choice
    if choice == "1":
        print("\nBuilding full-book RAG...")
        chain = build_rag(
            chapter_list=chapter_files,
            index_name="manualrag-full"
        )

    elif choice == "2":
        print("\nBuilding Chapter 2 RAG...")
        chain = build_rag(
            chapter_list=["ch02.txt"],
            index_name="manualrag-ch2"
        )

    elif choice == "3":
        print("\nBuilding Chapter 4 RAG...")
        chain = build_rag(
            chapter_list=["ch04.txt"],
            index_name="manualrag-ch4"
        )

    else:
        print("Invalid choice.")
        return

    # Ask a question
    query = input("\nEnter your question: ")
    result = chain.invoke(query)  #feed user input into selected RAG pipeline

    print("\nAnswer:\n", result)


if __name__ == "__main__":
    main()