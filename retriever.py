from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

VECTORSTORE_DIR = "vectorstore"
TOP_K = 4

def load_vectorstore():
    print("Loading embedding Model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )

    print(" Now loading VectorStore...")

    vectorstore = Chroma(
        persist_directory = VECTORSTORE_DIR,
        embedding_function = embedding_model
    )

    print("Vectorstore loaded Successfully........")
    return vectorstore


def build_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kargs = {"k": TOP_K}
    )

    return retriever

def retrieve_chunks(retriever, question):
    chunks = retriever.invoke(question)

    print(f"Found {len(chunks)} related Chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"---Chunk{i+1}---")
        print(f"Source : {chunk.metadata.get('source','unknown')}")
        print(f"Page   : {chunk.metadata.get('page','?')}")
        print(f"Content: {chunk.page_content[:200]}...")
        print()

    return chunks


def main():
    print("----retriever test----")

    vector = load_vectorstore()
    retrieve_data = build_retriever(vector)

    test_question = "What is the main contribution of this paper?"
    chunks = retrieve_chunks(retrieve_data, test_question)

    print(f"Retrieved {len(chunks)} chunks sucessfully")
    print("These Chunks will be passed to LLM in Phase 4...")


if __name__ == "__main__":
    main()