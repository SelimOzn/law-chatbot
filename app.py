import os
import sys
from langchain_chroma.vectorstores import Chroma
from sympy.physics.units import temperature
import streamlit as st
from db_conf import create_db
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import google.genai as genai
from google.genai import types
from sentence_transformers import SentenceTransformer

def load_db(db_path):
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
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
    client = genai.Client(api_key=API_KEY)
    config = types.GenerateContentConfig(
        temperature=0.5
    )
    db_path = "./data"
    vector_db = load_db(db_path)
    return client, config, vector_db, model_name


if __name__ == "__main__":
    st.set_page_config(page_title="Law Chatbot", page_icon="⚖️")
    st.title("⚖️ T.C. Hukuk Danışmanı Chatbot")
    st.markdown("Bu chatbot, `ipproo/Turkish-law` veri seti kullanılarak eğitilmiş bir RAG modelidir.\
                Yalnızca veri setinde bulunan bilgilerle cevap üretebilir.")
    client, config, vector_db, model_name = init_services()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query:= st.chat_input("Hukuki sorunuzu buraya yazın: "):
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append({"role":"user", "content":query})

        with st.chat_message("assistant"):
            with st.spinner("Yasal kaynaklar taranıyor, lütfen bekleyin..."):
                response = get_answer(client=client,
                                      config=config,
                                      vector_db = vector_db,
                                      model_name=model_name,
                                      query=query,
                                      top_k=3)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content":response})



