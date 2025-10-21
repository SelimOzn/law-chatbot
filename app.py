import os
import sys
from langchain_chroma.vectorstores import Chroma
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from db_conf import create_db
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document


# def get_initial_docs(vector_db):
#     return vector_db.get()["documents"]


def get_initial_docs(vector_db):
    print("Mevcut veritabanından dokümanlar BM25 için çekiliyor...")
    try:
        doc_strings = vector_db.get()["documents"]

        all_docs = [Document(page_content=s) for s in doc_strings]
        print(f"{len(all_docs)} adet doküman veritabanından başarıyla yüklendi.")
        return all_docs

    except Exception as e:
        st.error(f"ChromaDB'den dokümanlar alınırken hata oluştu: {e}")
        st.info("Veritabanının './data' klasöründe olduğundan emin olun.")
        st.stop()

def load_db(db_path):
    embed_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=embed_model_name)
    collection_name = "law_chatbot"

    if not os.path.exists(db_path):
        try:
            print("Database not found. Creating...")
            create_db(db_path)
        except Exception as e:
            print(f"Failed to create database: {e}")
            sys.exit(1)

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    return vectordb


def process_uploaded_files(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.getvalue())
            tmp_file_path = temp_file.name

        if file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file.type == "text/plain":
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        else:
            st.warning(f"Desteklenmeyen dosya tipi: {file.type}. Bu dosya atlandı.")
            os.remove(tmp_file_path)
            continue

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

        os.remove(tmp_file_path)

    return all_chunks


def get_rag_chain(vector_db, llm, docs):
    print("RAG zinciri oluşturuluyor...")

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    chroma_retriever = vector_db.as_retriever(search_kwargs={"k":5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.3,0.7]
    )

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    contextualize_q_system_prompt = """
        Given a chat history and the latest user question 
        which might reference context in the chat history, 
        formulate a standalone question which can be understood 
        without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
    Sen, T.C. Hukuk Sistemine dayalı, uzman bir hukuki danışman yapay zeka asistanısın. 
    Görevin, kullanıcının sorularını yalnızca sana sunulan bağlam (kaynak dokümanlar) dahilinde, 
    kesinlikle Türkçe hukuk terminolojisine uygun, net ve yasalara dayalı olarak yanıtlamaktır.

    [KURALLAR]
    1.  **Tek Kaynak:** Cevabını oluştururken KESİNLİKLE sadece sana "BAĞLAM" başlığı altında sağlanan bilgileri kullan. Bağlamda cevabın yoksa, "Size sunulan hukuki kaynaklarda bu konuya ilişkin kesin bir bilgi bulunmamaktadır." de.
    2.  **Hukuki Dil:** Yanıtlarında Türk hukukuna özgü doğru terimleri kullan.
    3.  **Yasa Maddesi:** Eğer bağlam sana yasa maddesi veya karar numarası sunuyorsa, cevabının sonunda ilgili kaynağı belirt.
    4.  **Tavsiye Değil:** Yasal tavsiye vermekten kaçın. Cevabını bir bilgilendirme olarak sun.

    [BAĞLAM]
    {context}

    [KONUŞMA GEÇMİŞİ]
    {chat_history}

    [KULLANICI SORUSU]
    {input}

    Yukarıdaki [KURALLAR], [BAĞLAM] ve [KONUŞMA GEÇMİŞİ]'ni dikkate alarak, [KULLANICI SORUSU]'na T.C. Hukuk Sistemi perspektifinden cevap ver.
    Yanıtınız:"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def get_answer(client, model_name, config, query, vector_db, top_k=3):
    retrieval_res = vector_db.similarity_search(query, k=top_k)
    context = ["\n\n".join([doc.page_content for doc in retrieval_res])]

    prompt = f"""
    Sen, T.C. Hukuk Sistemine dayalı, uzman bir hukuki danışman yapay zeka asistanısın. Görevin, kullanıcının sorularını yalnızca sana sunulan bağlam (kaynak dokümanlar) dahilinde, kesinlikle Türkçe hukuk terminolojisine uygun, net ve yasalara dayalı olarak yanıtlamaktır.
    
    [KURALLAR]
    1.  **Tek Kaynak:** Cevabını oluştururken KESİNLİKLE sadece sana "BAĞLAM" başlığı altında sağlanan bilgileri kullan. Bağlamda cevabın yoksa veya yetersizse, bunu dürüstçe ve diplomatik bir dille belirt (Örn: "Size sunulan hukuki kaynaklarda bu konuya ilişkin kesin bir bilgi bulunmamaktadır." veya "Bu konuda daha detaylı bilgi için bir avukata danışmanız önerilir."). Kendi genel bilginle ya da uydurma bilgilerle (halüsinasyon) cevap verme.
    2.  **Hukuki Dil:** Yanıtlarında Türk hukukuna özgü doğru terimleri (örneğin, 'İstinaf', 'Yargıtay', 'Temlik', 'Müeccel Borç', 'Zilyetlik', 'Tebligat' vb.) kullanmaya özen göster. Cevap akıcı ve resmi Türkçe ile yazılmalıdır.
    3.  **Yasa Maddesi:** Eğer bağlam sana yasa maddesi veya karar numarası sunuyorsa, cevabının sonunda ilgili maddeyi/kararı veya kaynağı (örneğin: "Türk Medeni Kanunu Madde 30", "Yargıtay 12. Hukuk Dairesi'nin X tarihli kararı" veya "Kaynağa göre...") kesinlikle belirt.
    4.  **Tavsiye Değil:** Yasal tavsiye vermekten kaçın. Cevabını bir bilgilendirme olarak sun. Asla kullanıcıyı vekâleten temsil etme, hukuki sonuca bağlama veya "kesinlikle şunu yapmalısınız" gibi yönlendirici ifadeler kullanma.
    5.  **Biçimlendirme:** Cevabını paragraflar ve, uygun olduğu yerlerde, maddeler halinde düzenleyerek okunabilirliği artır.
    
    [BAĞLAM]
    {context}
    
    [KULLANICI SORUSU]
    {query}
    
    Yukarıdaki [KURALLAR] ve [BAĞLAM]'ı dikkate alarak, [KULLANICI SORUSU]'na T.C. Hukuk Sistemi perspektifinden, hukuki terimlerle ve kaynak belirterek cevap ver.
    
    Yanıtınız:"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config
        )
    return response.text

@st.cache_resource
def init_services():
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        st.error("Google API key is missing")
        st.stop()

    model_name = "gemini-2.0-flash"
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5, google_api_key=API_KEY)

    embed_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=embed_model)

    db_path = "./data"
    vector_db = load_db(db_path)

    if "all_docs" not in st.session_state:
        st.session_state.all_docs = get_initial_docs(vector_db)

    st.session_state.vector_db = vector_db
    st.session_state.llm = llm
    st.session_state.embedding_function = embedding_function
    st.session_state.all_docs_count = len(st.session_state.all_docs)

    return llm, vector_db, embedding_function


if __name__ == "__main__":
    st.set_page_config(page_title="Law Chatbot", page_icon="⚖️")
    st.title("⚖️ T.C. Hukuk Danışmanı Chatbot")
    st.markdown("Bu chatbot, `ipproo/Turkish-law` veri seti kullanılarak eğitilmiş bir RAG modelidir.\
                Yalnızca veri setinde bulunan bilgilerle cevap üretebilir.")

    if "llm" not in st.session_state:
        init_services()

    llm = st.session_state.llm
    vector_db = st.session_state.vector_db
    embedding_function = st.session_state.embedding_function

    if "rag_chain" not in st.session_state or st.session_state.all_docs_count != len(st.session_state.all_docs):
        st.session_state.rag_chain = get_rag_chain(vector_db, llm, st.session_state.all_docs)
        st.session_state.all_docs_count = len(st.session_state.all_docs)


    with st.sidebar:
        st.title("Doküman yönetimi")
        st.markdown("Veritabanına kendi hukuki belgelerinizi (.pdf/.txt) ekleyin.")
        uploaded_files = st.file_uploader(
            "Dosyaları buraya sürükleyin",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = set()
            new_files_to_process = []
            for file in uploaded_files:
                if file.name not in st.session_state.processed_files:
                    new_files_to_process.append(file)

            if new_files_to_process:
                with st.spinner(f"{len(new_files_to_process)} adet dosya işleniyor ve veritabanına ekleniyor..."):
                    new_chunks = process_uploaded_files(new_files_to_process)
                    if new_chunks:
                        st.session_state.vector_db.add_documents(new_chunks)
                        st.session_state.all_docs.extend(new_chunks)
                        st.session_state.rag_chain = get_rag_chain(
                            st.session_state.vector_db,
                            st.session_state.llm,
                            st.session_state.all_docs
                        )
                        st.session_state.all_docs_count = len(st.session_state.all_docs)
                        for file in new_files_to_process:
                            st.session_state.processed_files.add(file.name)

                        st.success(f"{len(new_chunks)} adet doküman parçası veri tabanına eklendi.")
                    else:
                        st.error("Dosyalar işlenemedi veya desteklenmeyen formatta.")

    history = StreamlitChatMessageHistory(key="messages")

    if len(history.messages) == 0:
        history.add_ai_message("Size T.C. Hukuk Sistemi kapsamında nasıl yardımcı olabilirim?")

    for msg in history.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)


    if query:= st.chat_input("Hukuki sorunuzu buraya yazın: "):
        st.chat_message("user").markdown(query)
        history.add_user_message(query)

        with st.chat_message("assistant"):
            with st.spinner("Yasal kaynaklar taranıyor, lütfen bekleyiniz..."):
                response = st.session_state.rag_chain.invoke(
                    {
                        "input":query,
                        "chat_history" : history.messages
                    }
                )
                answer = response.get('answer', 'Bir hata oluştu.')
                st.write(answer)

                with st.expander("Kullanılan hukiki kaynaklar"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.info(f"Kaynak {i+1}: \n\n{doc.page_content[:500]}...")

        history.add_ai_message(answer)