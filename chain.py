import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_llm():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature= 0.2,
        max_tokens=1024
    )

    print("Llm Mpodel Loaded...")
    return llm

def build_prompt():
    template = """
    You are ResearchMind, an expert AI assistant that answers questions
    about research papers with precision and clarity.

    Use ONLY the context provided below to answer the question.
    If the answer is not in the context, say "I couldn't find that 
    information in the provided document."

    Do NOT make up information. Do NOT use your training knowledge
    to fill in gaps. Only use what is in the context.

    Always end your answer by mentioning which page(s) the 
    information came from.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["context","question"]
    )

    return prompt

def format_chunks_from_context(chunks):
    context_parts = []

    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page","unknown")
        source = chunk.metadata.get("source","unknown")
        text = chunk.page_content.strip()

        context_parts.append(
            f"[Chunk {i+1} | Page: {page} | Source: {source} ] \n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return context

def rag_chain(retriever):
    llm = load_llm()
    prompt = build_prompt()
    output_parser = StrOutputParser()

    rag_chain = (
        {
            "context": retriever | (lambda chunks : format_chunks_from_context(chunks)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | output_parser
    )

    return rag_chain

def ask(rag_chain, question):
     print(f"\nQuestion: {question}")
     print("Thinking....\n")

     answer = rag_chain.invoke(question)
     print(f"Answer: {answer}")

     return answer


def main():

    from retriever import load_vectorstore, build_retriever

    vectorstore = load_vectorstore()
    chunks = build_retriever(vectorstore)

    chain = rag_chain(chunks)
    test_questions = [
        "What is the name of the author?",
        "What dataset was used for evaluation?",
        "What are the limitations mentioned by the authors?"
    ]

    for question in test_questions:
        ask(chain,question)
        print("\n"+ "="*50 +"\n")


if __name__ == "__main__":
    main()
