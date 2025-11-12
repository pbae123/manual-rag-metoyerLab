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

def main():
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    CHAPTERS_DIR = "data/chapters"
    
    # --- Load and concatenate all chapters ---
    all_texts = []
    chapter_files = sorted(
        [f for f in os.listdir(CHAPTERS_DIR) if f.endswith(".txt")]
    )
    
    if not chapter_files:
        print("❌ No chapter files found in data/chapters/")
        return
    
    print(f"📚 Loading {len(chapter_files)} chapters...\n")
    for file in chapter_files:
        chapter_path = os.path.join(CHAPTERS_DIR, file)
        with open(chapter_path, "r", encoding="utf-8") as f:
            text = f.read()
            all_texts.append(text)
            print(f"✅ Loaded {file} ({len(text)} characters)")
    
    full_text = "\n".join(all_texts)
    print(f"\n📘 Combined text length: {len(full_text)} characters\n")
    
    # --- Model & parser setup ---
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    
    # --- Prompt template ---
    template = """
    Answer the question based on the following context. 
    If you can't find the answer, just say that you don't know.
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # --- Split into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    print(f"🔹 Split into {len(chunks)} total chunks\n")
    
    # --- Embeddings ---
    embeddings = OpenAIEmbeddings()
    
    # --- Pinecone setup ---
    index_name = "manualrag"
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created new Pinecone index: {index_name}")
    else:
        print(f"🔗 Using existing Pinecone index: {index_name}")
    
    # --- Store all chapter chunks in Pinecone ---
    vectorstore = PineconeVectorStore.from_texts(chunks, embeddings, index_name=index_name)
    
    # --- Build retriever ---
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # --- Build RAG chain ---
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    
    # --- Ask a question ---
    query = input("\n🔎 Enter your question: ")
    result = chain.invoke(query)
    print("\n💬 Answer:\n", result)

if __name__ == "__main__":
    main()