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
    #allows you to access environment variables
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    CHAPTERS_DIR = "data/chapters"
    
    all_texts = []  #empty list that will later hold content of each chapter file

    chapter_files = sorted( #alphabetically sorting chapter names 
        [f for f in os.listdir(CHAPTERS_DIR) if f.endswith(".txt")]
    )
    
    if not chapter_files:  #if no chapter files 
        print("No chapter files found in data/chapters/")
        return
    
    for file in chapter_files: #looping over list of filenames 
        chapter_path = os.path.join(CHAPTERS_DIR, file) #creates full path to file 
        with open(chapter_path, "r", encoding="utf-8") as f:
            text = f.read()  #read file as single string 
            all_texts.append(text)  #add it to empty list, will contain all chapters at the end 
            print(f"Loaded {file} ({len(text)} characters)")
    
    full_text = "\n".join(all_texts) #just combining everything into one text block
    print(f"\n Combined text length: {len(full_text)} characters\n") 
    
    #setup model and parser 
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser() #converts response into a clean string, readable 
    
    #prompt template
    #instructions: how to answer 
    #context: retrieved text from pdf 
    #question: user's query 
    template = """
    Answer the question based on the following context. 
    If you can't find the answer, just say that you don't know.
    
    Context: {context}
    
    Question: {question}
    """
    #automatically converts template into format that will be sent to OpenAI model 
    prompt = ChatPromptTemplate.from_template(template)  
    
    #chunking 
    #defines how to split text into overlapping parts 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #actually splits the document 
    chunks = text_splitter.split_text(full_text)
    print(f"Split into {len(chunks)} total chunks\n")
    
    #embeddings
    embeddings = OpenAIEmbeddings()
    
    #pinecone setup 
    index_name = "manualrag"
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")
    
    #store all chapter chunks in Pinecone 
    vectorstore = PineconeVectorStore.from_texts(chunks, embeddings, index_name=index_name)
    
    # Build retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # Build RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    
    # Ask a question
    query = input("\n Enter your question: ")
    result = chain.invoke(query)
    print("\n Answer:\n", result)

if __name__ == "__main__":
    main()