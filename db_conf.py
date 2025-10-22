import os.path
import shutil
from datasets import load_dataset
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def create_db(db_path):
    embed_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Veri setini indirip uygun soru cevap formatına getirme işlemi yapılıyor
    dataset = load_dataset('ipproo/Turkish-law')
    train_ds = dataset['train']
    df = train_ds.to_pandas()
    text = "Soru: " + df["soru"] + "\n" + "Cevap: " + df["cevap"]
    ds = text.to_list()

    # Her soru cevap çiftini Document nesnesi olarak saklanılıyor
    docs = [Document(page_content=chunk) for chunk in ds]

    # Google embedding modeli kullanılıyor
    embedding_function = HuggingFaceEmbeddings(
        model_name = embed_model_name
    )

    # Temiz kurulum için vektör veri tabanı varsa siliniyor.
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception as e:
            print(f"Failed to remove existing {db_path}: {e}")

    # Vektör veri tabanı oluşturuluyor
    vector_db = Chroma.from_documents(
        docs,
        embedding=embedding_function,
        persist_directory=db_path,
        collection_name="law_chatbot"
    )

    return vector_db


if __name__ == '__main__':
    create_db("./data")

