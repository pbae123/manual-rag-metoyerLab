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


def build_rag(chapter_list, index_name):
    """
    Build a RAG system for a specific set of chapters.

    chapter_list: list of chapter filenames (e.g., ["chapter2.txt"])
    index_name: name of Pinecone index to use (keeps each RAG system separate)

    Returns:
        A LangChain RAG pipeline (retriever + prompt + model + parser)
    """

    #allows you to access environment variables
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    CHAPTERS_DIR = "data/chapters"

    #load only the specified chapters
    texts = []  # empty list to hold the contents of the selected chapters

    for chapter_file in chapter_list:
        chapter_path = os.path.join(CHAPTERS_DIR, chapter_file)

        if not os.path.exists(chapter_path):
            print(f"Warning: {chapter_path} does not exist, skipping.")
            continue

        with open(chapter_path, "r", encoding="utf-8") as f:
            text = f.read()  # read entire chapter as a single string
            texts.append(text)
            print(f"Loaded {chapter_file} ({len(text)} characters)")

    #combine all selected chapters into one block of text
    full_text = "\n".join(texts)
    print(f"\nCombined text length: {len(full_text)} characters\n")

    #chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # max size of each chunk
        chunk_overlap=200  # overlap ensures context continuity
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Split into {len(chunks)} total chunks\n")

    #embeddings
    embeddings = OpenAIEmbeddings()

    #pinecone setup
    pc = PineconeClient(api_key=PINECONE_API_KEY)

    #create index if it doesn't already exist
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")

    #store the document chunks into the Pinecone index
    vectorstore = PineconeVectorStore.from_texts(
        chunks,
        embeddings,
        index_name=index_name
    )

    #build retriever for similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # top 10 most similar chunks
    )

    #prompt template (instructions + context + user question)
    template = """
    Answer the question based on the following context. 
    If you can't find the answer, just say that you don't know.
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    #model and parser
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()

    #rag chain pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt  # fills the template
        | model   # sends final prompt to OpenAI
        | parser  # creates clean output string
    )

    return chain
