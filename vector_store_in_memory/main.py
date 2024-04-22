import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

if __name__ == "__main__":
    print("Hello, World!")

    current_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(current_dir, "react_synergizing_reasoning_an.pdf")

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=30,
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()

    # Create the vector store
    persist_directory = os.path.join(current_dir, "chroma_db_local")
    vector_store = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=persist_directory
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=vector_store.as_retriever()
    )
    res = qa_chain.run("Give me the gist of ReAct in 3 sentences")

    print(res)
