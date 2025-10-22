<div align="center">

# T.C. Hukuk Danışmanı Chatbot



T.C. Hukuk Sistemi üzerine uzmanlaşmış, RAG (Retrieval-Augmented Generation) tabanlı yapay zeka chatbot. `ipproo/Turkish-law` veri seti ve kullanıcı tarafından yüklenen PDF/TXT dosyaları üzerinden hukuki soruları yanıtlar.

</div>

## Kullanımı

-   Chatbot'a doğrudan hukuki sorular sorabilirsiniz (Örn: "Federasyon nasıl kurulur?").
-   Spesifik madde numaralarını sorgulayabilirsiniz (Örn: "Medeni Kanun Madde 96 nedir?").
-   Kenar çubuğundaki (sidebar) "Doküman Yönetimi" alanından kendi `.pdf` veya `.txt` uzantılı hukuki belgelerinizi (kanun metinleri, Yargıtay kararları vb.) yükleyebilirsiniz.
-   Chatbot, hem temel veri setinden hem de sizin yüklediğiniz belgelerden (Hybrid Search kullanarak) kaynak bularak cevap üretir.
-   Konuşma hafızası sayesinde, önceki mesajlarınıza atıfta bulunarak takip soruları (Örn: "Peki bu davanın temyizi nedir?") sorabilirsiniz.
-   Cevabın altında "Kullanılan hukuki kaynaklar" bölümünden, cevabın hangi belgelere dayandığını görebilirsiniz.

## Teknoloji

-   **Python 3.10+**
-   **Streamlit:** Web arayüzü
-   **LangChain:** RAG pipeline'ı, zincirler (chains), hafıza (memory) ve retrieval (veri getirme) yönetimi
-   **Google Gemini (GenAI API):** Dil modeli (LLM)
-   **ChromaDB:** Vektör veritabanı (semantik arama için)
-   **BM25 (rank-bm25):** Anahtar kelime (keyword) araması
-   **EnsembleRetriever (LangChain):** Semantik (Chroma) ve anahtar kelime (BM25) aramalarını birleştiren Hybrid Search (Hibrit Arama)
-   **Hugging Face Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` modeli (metinleri vektöre dönüştürmek için)
-   **Hugging Face Datasets:** `ipproo/Turkish-law` (temel bilgi veri seti)

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Depoyu klonlayın:**
    ```bash
    # KULLANICI-ADI/PROJE-ADI kısımlarını kendi depo bilgilerinizle değiştirin
    git clone [https://github.com/KULLANICI-ADI/PROJE-ADI.git](https://github.com/KULLANICI-ADI/PROJE-ADI.git)
    cd PROJE-ADI
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Anahtarını ayarlayın:**
    -   Proje ana dizininde `.env` adında bir dosya oluşturun.
    -   İçine Google GenAI (MakerSuite) üzerinden aldığınız API anahtarınızı ekleyin:
      ```
      GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      ```

4.  **Vektör Veritabanını Oluşturun:**
    -   Temel `ipproo/Turkish-law` veri setini kullanarak ilk veritabanını oluşturmak için `db_conf.py` betiğini çalıştırın.
    -   Bu işlem `./data` adında bir klasör oluşturacaktır.
    ```bash
    python db_conf.py
    ```

5.  **Uygulamayı çalıştırın:**
    ```bash
    streamlit run app