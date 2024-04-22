import os
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Hello, World!")

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "react_synergizing_reasoning_an.pdf")
    loader = PyPDFLoader(file_path=file_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=30,
    )

    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local("faiss_index_react")

    new_vector_store = FAISS.load_local(
        "faiss_index_react", embeddings=embeddings, allow_dangerous_deserialization=True
    )
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vector_store.as_retriever()
    )
    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)
