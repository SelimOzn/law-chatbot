import os.path
import shutil
from datasets import load_dataset
import google.genai as genai
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain_core.documents import Document

def create_db(db_path):
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dataset = load_dataset('ipproo/Turkish-law')
    train_ds = dataset['train']
    df = train_ds.with_format("pandas")
    text = "Soru: " + df["soru"] + "\n" + "Cevap: " + df["cevap"]
    ds = text.to_list()
    docs = [Document(page_content=chunk) for chunk in ds]

    embedding_function = HuggingFaceEmbeddings(
        model_name = embed_model_name
    )

    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception as e:
            print(f"Failed to remove existing {db_path}: {e}")

    vector_db = Chroma.from_documents(
        docs,
        embedding=embedding_function,
        persist_directory=db_path,
        collection_name="law_chatbot"
    )

    return vector_db





if __name__ == '__main__':
    create_db("asfa")

