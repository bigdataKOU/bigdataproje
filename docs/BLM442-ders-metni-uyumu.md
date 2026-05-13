# BLM442 — Ders proje metni (`Açıklama.txt`) ile repo uyumu

Bu dosya, ders materyalindeki **Dönem Projesi** açıklaması ile **bu repoda** yapılanların karşılaştırmasıdır. Durumlar: **Yapıldı** | **Kısmen / süreç** | **Yapılması gereken (takım)**.

---

## 1. Teknik beklenti (açıklama metninin ilk paragrafı)

Metinde geçen: *Docker ile konteynerize ortam; Kafka ile streaming veri üretimi; Spark ile veri işleme; Delta Lake ile depolama; makine öğrenmesi ile tahmin; MLflow ile deney takibi; uçtan uca pipeline.*

| Madde | Projede | Durum |
|--------|---------|--------|
| Docker ile konteynerize ortam | `docker-compose.yml`, servisler (`kafka`, `spark-master`, `spark-worker`, `mlflow`, `producer`, `pipeline`, `dashboard`) | **Yapıldı** |
| Kafka ile streaming veri üretimi | Producer → `ratings` topic; Apache Kafka (KRaft) | **Yapıldı** |
| Apache Spark ile veri işleme | PySpark: bronze ingest (streaming), silver/gold batch, ALS train, inference | **Yapıldı** |
| Delta Lake ile depolama | Bronze / Silver / Gold katmanları, `delta-data` volume | **Yapıldı** |
| Makine öğrenmesi ile tahmin | ALS (MovieLens), batch öneriler (`user_recommendations`) | **Yapıldı** |
| MLflow ile deney takibi | Tracking URI, run parametreleri/metrikleri, model registry | **Yapıldı** |
| Uçtan uca büyük veri pipeline’i | `scripts/run_all.sh`, Makefile hedefleri, zincir akış | **Yapıldı** |

**Not:** ALS eğitimi ve inference’ın yerel artifact’larla güvenilir çalışması için `SPARK_MASTER_URL=local[*]` ile çalıştırma (`run_all.sh`, `make train` / `make inference`) tercih edilir; açıklama metninde ayrıca yazmıyor, proje içi pratik olarak eklendi.

---

## 2. İdari ve süreç kuralları (açıklama maddeleri)

| Madde | Açıklama | Projede / takımda | Durum |
|--------|-----------|-------------------|--------|
| Grup 3–4 kişi | Zorunlu | `README.md` Takım tablosu (4 kişi) | **Yapıldı** (içerik olarak) |
| En geç **5 Mayıs 2026** — grup, konu, **form** | Google Form bağlantısı metinde | Formun doldurulması | **Yapılması gereken** (takım; tarih ders koşuluna bağlı) |
| Her grup **farklı veri seti**; aynı veriden en fazla **2 grup** | İlk gelen alır | MovieLens 25M seçimi + hocaya/form uyumu | **Kısmen** — veri seti seçilmiş; **form/onay** takımın doğrulaması |
| Kodların tamamı **GitHub**’da, **commit geçmişi** değerlendirilir | — | Repo `bigdataKOU/bigdataproje`, anlamlı commit alışkanığı | **Kısmen** — **push + commit disiplini** sürekli |
| İntihal / kod kopyalama | Tüm grup notu sıfır | Özgün kullanım, kaynak gösterme | **Sürekli sorumluluk** |
| Üyelerin **eşit katkı**sı; sunumda **bireysel soru**lar | — | Görev dağılımı, herkesin mimariyi bilmesi | **Yapılması gereken** (sunum öncesi) |
| **Databricks CE** veya **yerel Spark** kabul | — | Docker + yerel Spark (PySpark) | **Yapıldı** (yerel/konteyner yolu) |
| **Sunumlar** (14 Mayıs, 21 Mayıs, 4 Haziran, **11 Haziran 2026**) | Ders saatinde | Slidelar, demo, soru-cevap hazırlığı | **Yapılması gereken** |

---

## 3. Repo dokümantasyonu ile gerçek mimari

`README.md` içinde bazı servis tablolarında **Bitnami** image örnekleri geçebilir; güncel `docker-compose.yml` **Apache Kafka / Apache Spark** tabanlıdır. Sunum veya raporda **gerçek compose** ile tabloyu hizalamak iyi olur.

| Madde | Durum |
|--------|--------|
| README servis tablosunun güncel image’larla uyumu | **Yapılması gereken** (küçük dokü güncellemesi) |

---

## 4. Kısa özet

- **Ders metnindeki teknik çerçeve** (Docker, Kafka, Spark, Delta, ML, MLflow, uçtan uca akış) bu projede **karşılanmış** sayılır.
- **Metindeki idari maddeler** (form, sunum, eşit katkı, commit geçmişi, veri seti kuralı) **çoğu takım süreci**; repoda otomatik doğrulanmaz.
- Teknik tarafta bilinen ek: **ALS train/inference** için `local[*]` kullanımı ve `fixes.txt` / `run_all` notları; ders PDF’inde yok, **proje güvenilirliği** için eklendi.

---

*Son güncelleme: 2026 — BLM442 Büyük Veri Analizine Giriş, dönem projesi.*


