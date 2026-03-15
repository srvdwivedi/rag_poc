from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_DB_DIR


def ingest_documents():

    print("Loading documents...")

    loader = PyPDFLoader("data/sample_rag_test.pdf")
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} chunks")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectorstore.persist()

    print("Vector DB created successfully")


if __name__ == "__main__":
    ingest_documents()