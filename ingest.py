import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()
PDF_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

def load_documents(pdf_dir):
    all_documents = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            print(f"Filename: {filename}")

            loader = PyPDFLoader(filepath)
            documents = loader.load()

            all_documents.extend(documents)
    
    print("Total pages loaded: ",len(all_documents))
    return all_documents


def text_splitter(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n","\n","."," "]
    )

    chunks = splitter.split_documents(documents)

    print("Chunks size",len(chunks))
    return chunks

def embed_and_store(chunks, vectorstore_dir):

    print("Loading embedding model(first time may take a minute)...")

    embedding_model = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )

    print("Embedding chunks and Storing data To vectordatabase in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = vectorstore_dir
    )

    print(f"Done! {len(chunks)} saved to ChromaDB")
    return vectorstore


def main():
    
    documents = load_documents(PDF_DIR)

    chunks = text_splitter(documents)

    embed_and_store(chunks, VECTORSTORE_DIR)


if __name__ == "__main__":
    main()