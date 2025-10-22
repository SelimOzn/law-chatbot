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

# Veri tabanına eklenen tüm dokümanlar, BM25 retreiver için getirir
def get_initial_docs(vector_db):
    print("Mevcut veri tabanından dokümanlar BM25 için çekiliyor...")
    try:
        # Veri tabanındaki tüm dokümanların ham metin halini alır
        doc_strings = vector_db.get()["documents"]

        all_docs = [Document(page_content=s) for s in doc_strings]
        print(f"{len(all_docs)} adet doküman veri tabanından başarıyla yüklendi.")
        return all_docs

    # Veri tabanı yoksa veya erişim hatası varsa Streamlit üzerinden kullanıcıya hata gösterir
    except Exception as e:
        st.error(f"ChromaDB'den dokümanlar alınırken hata oluştu: {e}")
        st.info("Veri tabanının './data' klasöründe olduğundan emin olun.")
        st.stop()

# Vektör veri tabanını yükler, yoksa oluşturur
def load_db(db_path):
    embed_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=embed_model_name)
    collection_name = "law_chatbot"

    # Veri tabanı klasörü yoksa create_db() ile veri tabanı oluşturulur
    if not os.path.exists(db_path):
        try:
            print("Database not found. Creating...")
            create_db(db_path)
        except Exception as e:
            print(f"Failed to create database: {e}")
            sys.exit(1)

    # Vektör veri tabanı örneği oluşturulur
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    return vectordb

# Kullanıcının yüklediğin PDF veya TXT dosyalarını işler ve metin parçalarına ayırır
def process_uploaded_files(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        # Dosya yükleyicileri dosyanın yolunu beklediği için yüklenen dosyalar geçici olarak kaydedilir
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
        # Uzun metinleri RAG için uygun küçük parçalara (chunk) böler
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)
        # Geçici dosyayı sistemden siler
        os.remove(tmp_file_path)

    return all_chunks

# RAG zincirini kurar
def get_rag_chain(vector_db, llm, docs):
    print("RAG zinciri oluşturuluyor...")

    # Anahtar kelime tabanlı retreiver
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    # Embedding modelinin oluşturduğu vektörler için retreiver
    chroma_retriever = vector_db.as_retriever(search_kwargs={"k":5})
    # BM25 ve Chroma vektör sonuçlarını birleştir
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.3,0.7]
    )

    compressor = LLMChainExtractor.from_llm(llm)
    # LLM kullanarak getirilen metin parçalarından gereksiz bilgileri temizleyip ilgili kısımları tutar
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # Kullanıcının önceki konuşmalarını dikkate alarak soruyu bağımsız hale getirir
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question 
        which might reference context in the chat history, 
        formulate a standalone question which can be understood 
        without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),    # Konuşma geçmişinin verilmesi durumu için yer tutucu
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    # Modelin yanıt oluşturma kurallarını ve bağlamını tanımlar
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

    # Retreiver'dan gelen dokümanları alıp LLM'e göndererek cevap üretir
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Retreiver ve cevap üretme adımlarını birleştiren ana RAG zinciri
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


@st.cache_resource  # Bu fonksiyonun sonucunun önbelleğe alınıp uygulama yeniden başlatıldığında
                    # hafızadaki sonucun kullanılmasını sağlar
# LLM ve embedding modellerini başlatır, veri tabanı bağlantısını açar
def init_services():
    load_dotenv()   # .env dosyasındaki ortam değişkenlerini yükler
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        st.error("Google API key is missing")
        st.stop()

    model_name = "gemini-2.0-flash"
    # Gemini LLM modelini başlatır
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5, google_api_key=API_KEY)

    embed_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=embed_model)

    db_path = "./data"
    vector_db = load_db(db_path)

    # Veri tabanındaki tüm dokümanları yükleyip Streamlit oturum hafızasına kaydeder
    if "all_docs" not in st.session_state:
        st.session_state.all_docs = get_initial_docs(vector_db)

    # Gerekli nesneler oturum hafızasına kaydedilir
    st.session_state.vector_db = vector_db
    st.session_state.llm = llm
    st.session_state.embedding_function = embedding_function
    st.session_state.all_docs_count = len(st.session_state.all_docs)

    return llm, vector_db, embedding_function


if __name__ == "__main__":
    # Streamlit arayüz başlığı ve ikon ayarları
    st.set_page_config(page_title="Law Chatbot", page_icon="⚖️")
    st.title("⚖️ T.C. Hukuk Danışmanı Chatbot")
    st.markdown("Bu chatbot, `ipproo/Turkish-law` veri seti kullanılarak eğitilmiş bir RAG modelidir.\
                Yalnızca veri setinde bulunan bilgilerle cevap üretebilir.")

    # Uygulama ilk kez açılıyorsa veya cache sıfırlandıysa servisler başlatılır ve oturum hafızasına alınır
    if "llm" not in st.session_state:
        init_services()

    # Gerekli nesneler oturum hafızasından alınır
    llm = st.session_state.llm
    vector_db = st.session_state.vector_db
    embedding_function = st.session_state.embedding_function

    # RAG zinciri hafızada yoksa veya yeni doküman eklendiği için doküman sayısı değiştiyse zincir oluşturulur ve
    # oturum hafızasına kaydedilir
    if "rag_chain" not in st.session_state or st.session_state.all_docs_count != len(st.session_state.all_docs):
        st.session_state.rag_chain = get_rag_chain(vector_db, llm, st.session_state.all_docs)
        st.session_state.all_docs_count = len(st.session_state.all_docs)

    # Kullanıcının PDF/TXT yüklemesi için yan panel oluşturur
    with st.sidebar:
        st.title("Doküman yönetimi")
        st.markdown("Veri tabanına kendi hukuki belgelerinizi (.pdf/.txt) ekleyin.")

        # Dosya yükleme bileşeni oluşturur ve değişkene atar
        uploaded_files = st.file_uploader(
            "Dosyaları buraya sürükleyin",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        # Kullanıcı dosya yüklediyse yüklenen dosyaların işlenmesi yapılır
        if uploaded_files:
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = set()
            new_files_to_process = []
            # Sadece oturum hafızasında olmayan dosyaların işlenmesini sağlar
            for file in uploaded_files:
                if file.name not in st.session_state.processed_files:
                    new_files_to_process.append(file)

            if new_files_to_process:
                with st.spinner(f"{len(new_files_to_process)} adet dosya işleniyor ve veri tabanına ekleniyor..."):
                    # Yeni eklenen dokümanı işler
                    new_chunks = process_uploaded_files(new_files_to_process)
                    if new_chunks:
                        st.session_state.vector_db.add_documents(new_chunks)
                        st.session_state.all_docs.extend(new_chunks)
                        # Yeni eklenen dokümanlarla RAG zincirini yeniden oluşturur (BM25'den dolayı)
                        st.session_state.rag_chain = get_rag_chain(
                            st.session_state.vector_db,
                            st.session_state.llm,
                            st.session_state.all_docs
                        )
                        st.session_state.all_docs_count = len(st.session_state.all_docs)
                        # Yeni eklenen dokümanları, doküman geçmişine ekler
                        for file in new_files_to_process:
                            st.session_state.processed_files.add(file.name)

                        st.success(f"{len(new_chunks)} adet doküman parçası veri tabanına eklendi.")
                    else:
                        st.error("Dosyalar işlenemedi veya desteklenmeyen formatta.")

    # Sohbet geçmişini, st.session_state.messages adlı değişkene bağlar. History üzerindeki her değişiklik
    # st.session_state'e otomatik yansır.
    history = StreamlitChatMessageHistory(key="messages")

    # İlk asistan mesajını oluşturur
    if len(history.messages) == 0:
        history.add_ai_message("Size T.C. Hukuk Sistemi kapsamında nasıl yardımcı olabilirim?")

    # Hafızadaki tüm mesajları ekrana yazdırır
    for msg in history.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # Kullanıcının yazdığı yeni mesajı alır
    if query:= st.chat_input("Hukuki sorunuzu buraya yazın: "):
        st.chat_message("user").markdown(query)
        # Yeni mesajı, mesaj geçmişine ekler
        history.add_user_message(query)

        with st.chat_message("assistant"):
            with st.spinner("Yasal kaynaklar taranıyor, lütfen bekleyiniz..."):
                # Kullanıcı mesajını RAG zincirine verir ve LLM modelinden alınan cevabı döndürür
                response = st.session_state.rag_chain.invoke(
                    {
                        "input":query,
                        "chat_history" : history.messages
                    }
                )
                answer = response.get('answer', 'Bir hata oluştu.')
                st.write(answer)

                # LLM'in cevap üretirken bağlam olarak kullandığı gerçek doküman parçalarını kullanıcıya gösterir
                with st.expander("Kullanılan hukiki kaynaklar"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.info(f"Kaynak {i+1}: \n\n{doc.page_content[:500]}...")

        # Asistanın cevabını mesaj geçmişine ekler
        history.add_ai_message(answer)