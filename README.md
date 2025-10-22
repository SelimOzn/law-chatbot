<div align="center">

# T.C. Hukuk Danışmanı Chatbot


T.C. Hukuk Sistemi üzerine uzmanlaşmış, RAG (Retrieval-Augmented Generation) tabanlı yapay zeka chatbot. `ipproo/Turkish-law` veri seti ve kullanıcı tarafından yüklenen PDF/TXT dosyaları üzerinden hukuki soruları yanıtlar.

</div>

## Kullanımı

-   Chatbot'a doğrudan hukuki sorular sorabilirsiniz (Örn: "Federasyon nasıl kurulur?").
-   Kenar çubuğundaki (sidebar) "Doküman Yönetimi" alanından kendi `.pdf` veya `.txt` uzantılı hukuki belgelerinizi (kanun metinleri, Yargıtay kararları vb.) yükleyebilirsiniz.
-   Chatbot, hem temel veri setinden hem de sizin yüklediğiniz belgelerden (Hybrid Search kullanarak) kaynak bularak cevap üretir.
-   Konuşma hafızası sayesinde, önceki mesajlarınıza atıfta bulunarak takip soruları (Örn: "Peki bu davanın temyizi nedir?") sorabilirsiniz.
-   Cevabın altında "Kullanılan hukuki kaynaklar" bölümünden, cevabın hangi belgelere dayandığını görebilirsiniz.

## Veri Seti

-   ipproo/Turkish-law; yalnızca Türk anayasasını içeren, Chatgpt, Claude vb. gibi sohbet robotları kullanılarak toplanmış, soru cevap şeklinde veriler içerir.
-   Eğer kullanıcı doküman yüklerse, doğrudan bu doküman da veri seti olarak kullanılmaktadır.

## Teknoloji

-   **Python 3.12**
-   **Streamlit:** Web arayüzü
-   **LangChain:** RAG pipeline'ı, zincirler (chains), hafıza (memory) ve retrieval (veri getirme) yönetimi
-   **Google Gemini (langchain-google-genai):** Dil modeli (LLM)
-   **ChromaDB:** Vektör veritabanı (semantik arama için)
-   **BM25 (rank-bm25):** Anahtar kelime (keyword) araması
-   **EnsembleRetriever (LangChain):** Semantik (Chroma) ve anahtar kelime (BM25) aramalarını birleştiren Hybrid Search (Hibrit Arama)
-   **Hugging Face Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` modeli (metinleri vektöre dönüştürmek için)
-   **Hugging Face Datasets:** `ipproo/Turkish-law` (temel bilgi veri seti)

## RAG Mimarisi (RAG Architecture)

Bu proje, bir sorguyu yanıtlamak için gelişmiş, hafıza destekli bir RAG (Retrieval-Augmented Generation) mimarisi kullanır. Sistem, temel olarak 4 adımdan oluşur:

### 1. Konuşma Hafızası ve Sorgu Yeniden Yazma (Query Rewriting)

Kullanıcının yeni sorgusu (`input`) ve tüm konuşma geçmişi (`chat_history`) alınır. Bu bilgiler, `create_history_aware_retriever` zinciri kullanılarak LLM'e (Gemini) gönderilir. LLM, geçmiş konuşmaya dayalı belirsiz sorguları (örn: "Peki bu davanın temyizi?") tek başına anlamlı bir arama sorgusuna (örn: "Boşanma davasının temyiz süreci nedir?") dönüştürür.

### 2. Veri Getirme - Hibrit Arama (Hybrid Search)

Yeniden yazılan bu sorgu, `EnsembleRetriever`'a (Hibrit Arayıcı) gönderilir. Bu arayıcı, en doğru bağlamı bulmak için iki farklı arama yöntemini eş zamanlı olarak çalıştırır:

* **Anahtar Kelime Araması (BM25):** `BM25Retriever` kullanarak hafızadaki tüm dokümanlarda (yüklenen dosyalar dahil) anahtar kelime eşleşmesi arar. Bu, "Madde 96" veya "Konfederasyon" gibi spesifik terimler için etkilidir.
* **Semantik Arama (ChromaDB):** Sorguyu `paraphrase-multilingual-mpnet-base-v2` modeli ile vektöre dönüştürür ve ChromaDB veritabanında anlamsal olarak en yakın doküman parçalarını arar. Bu, "mal paylaşımı nasıl yapılır?" gibi kavramsal sorgular için etkilidir.

İki arama yönteminin sonuçları, belirlenen ağırlıklara (`weights=[0.3, 0.7]`) göre birleştirilerek en alakalı doküman parçaları (`context`) seçilir.

### 3. Cevap Üretme (Generation)

Bulunan bu doküman parçaları (`{context}`), tam konuşma geçmişi (`{chat_history}`) ve kullanıcının *orijinal* sorusu (`{input}`), `qa_system_prompt` adı verilen detaylı bir sistem talimatına yerleştirilir. Bu prompt, LLM'e (Gemini) sıkı `[KURALLAR]` (örn: "Sadece sana sunulan bağlamı kullan", "Yasal tavsiye verme") verir. LLM, bu zenginleştirilmiş prompt'u işler ve kurallara bağlı kalarak nihai cevabı üretir.

### 4. Dinamik Belge Yükleme (Dynamic Document Ingestion)

Kullanıcı kenar çubuktan (sidebar) yeni bir `.pdf` veya `.txt` dosyası yüklediğinde:
1.  Belge `RecursiveCharacterTextSplitter` ile 2000 karakterlik parçalara (`chunks`) bölünür.
2.  Bu yeni parçalar hem `ChromaDB` veritabanına (semantik arama için) hem de hafızadaki `all_docs` listesine (anahtar kelime araması için) eklenir.
3.  `BM25Retriever`, güncellenen `all_docs` listesi üzerinden **sıfırdan yeniden oluşturulmak** zorundadır.
4.  Bu nedenle, yeni dokümanları içermesi için tüm `rag_chain` (RAG zinciri) yeniden kurulur.

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Depoyu klonlayın:**
    ```bash
    git clone https://github.com/SelimOzn/law-chatbot.git
    cd law-chatbot
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    # Virtual environment oluşturup active edin (opsiyonel)
    python3 -m venv chatbot-env
    chatbot-env\Scripts\activate #Windows
    #source chatbot-env/bin/activate  # macOS/Linux
    pip install -r requirements.txt
    ```

3.  **API Anahtarını ayarlayın:**
    -   Proje ana dizininde `.env` adında bir dosya oluşturun.
    -   İçine Google AI Studio üzerinden aldığınız API anahtarınızı ekleyin:
      ```
      GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      ```

4.  **Vektör Veritabanını Oluşturun:**
    -   Uygulama, veritabanı yoksa ilk açılışta otomatik olarak oluşturabilir. Ancak bu işlem (embedding hesaplama) **birkaç dakika sürebilir** ve uygulamanın ilk açılışta donmuş gibi görünmesine neden olabilir.

    -   Bunu engellemek ve daha hızlı bir ilk açılış sağlamak için, uygulamayı çalıştırmadan *önce* aşağıdaki komutla veritabanını manuel olarak oluşturmanız önerilir:
    ```bash
    python db_conf.py
    ```

5.  **Uygulamayı çalıştırın:**
    ```bash
    streamlit run app.py

## Proje Yapısı
```
law-chatbot/
├── data/                     # ChromaDB veritabanı dosyalarının saklandığı klasör (db_conf.py tarafından oluşturulur)
├── app.py                    # Ana Streamlit uygulaması + RAG sorgu akışı
├── db_conf.py                # Veri tabanını oluşturma
├── requirements.txt          # Bağımlılıklar
├── .env                      # GOOGLE_API_KEY
└── README.md                 # Bu dosya
```

## Web Linki
- [https://huggingface.co/spaces/SelimOzn/law-chatbot](https://huggingface.co/spaces/SelimOzn/law-chatbot)

![Uygulama Arayüzü](images/website.png)
**Şekil 1**  Hugging Face Spaces üzerinde çalışan *T.C. Hukuk Danışmanı Chatbot* arayüzü. Solda kullanıcıların kendi hukuki belgelerini (.pdf veya .txt) yükleyebileceği doküman yönetimi bölümü, sağda ise chatbot ile etkileşim arayüzü görülüyor. Chatbot, `ipproo/Turkish-law` veri seti ve kullanıcılar tarafından yüklenen dokümanlar ile eğitilmiş bir RAG modeli olup, Türk hukuk sistemi kapsamında sorulara yanıt üretebiliyor. Kullanılan hukuki kaynaklar kısmında modelin cevap verirken bağlam olarak yararlandığı gerçek veri parçalarının özetlenmiş hallerini bulabilirsiniz. 
