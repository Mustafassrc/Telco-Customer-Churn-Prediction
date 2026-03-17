# Müşteri Kayıp (Churn) Tahmin Modeli

Bu proje, telekomünikasyon müşteri verilerini analiz ederek hangi müşterilerin aboneliklerini iptal edebileceğini (churn) makine öğrenimi algoritmaları kullanarak tahmin etmeyi amaçlayan bir veri bilimi çalışmasıdır.

## 📂 Proje Yapısı

```text
Customer_Churn_Prediction/
│
├── data/
│   ├── raw/                 # Orjinal ve işlenmemiş veriler
│   └── processed/           # Temizlenip mühendislik (feature engineering) işleminden geçmiş veriler
│
├── models/                  # Eğitilmiş (kaydedilmiş) makine öğrenimi modelleri ve ölçeklendiriciler (*.pkl)
│
├── notebooks/
│   └── 01_veri_kesfi.ipynb  # Veri analizi, temizleme, SMOTE ve model eğitim süreçlerini barındıran Jupyter Notebook
│
├── scripts/
│   └── tahmin_yap.py        # Kayıtlı modeli çağırarak rastgele veya sisteme girilen bir test verisiyle tahmin simülasyonu yapan betik
│
├── src/
│   └── pipeline.py          # Modeli ve yeni gelecek müşteri sözlüğünü alarak canlı tahmin üretebilen temel fonksiyon
│
├── README.md                # Proje bilgi dosyası
└── requirements.txt         # Projenin çalışması için gereken Python kütüphaneleri
```

## 🚀 Kurulum

1. **Python Ortamını Hazırlayın**  
   Projeyi çalıştırmak için Python 3.8 veya üzeri bir sürümün yüklü olduğundan emin olun.

2. **Gerekli Kütüphaneleri Yükleyin**  
   Proje dizinine terminal ile gidin ve aşağıdaki komutu çalıştırarak gerekli bağımlılıkları indirin:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Nasıl Çalışır?

#### 1. Veri Analizi ve Model Üretimi
Bütün veri analizleri, eksik veri doldurma, özellik mühendisliği (Feature Engineering) ve **Lojistik Regresyon** algoritmasının eğitilmesi işlemleri `notebooks/01_veri_kesfi.ipynb` dosyasının içinde gerçekleşmektedir. Bu dosyayı baştan sona çalıştırdığınızda oluşturulan modeller otomatik olarak `models/` klasörüne, işlenmiş temiz veriler ise `data/processed/` klasörüne kaydedilir.

#### 2. Model Testi (Canlı Simülasyon)
Modelin hazır olup olmadığını test etmek için `scripts/` klasörünün içindeki `tahmin_yap.py` betiği kullanılabilir.
Bu betik `data/processed/islenmis_test_verisi.csv` içerisinden rastgele bir müşteri satırı seçer, hedef değişkeni (Churn) modelden saklar ve modelin o müşteri için ayrılma tahmini üretmesini sağlar.

**Çalıştırmak için:**
Terminalinizde (PS veya CMD) proje ana dizininde olduğunuzdan emin olun ve şu komutları sırasıyla girin:
```bash
cd scripts
python tahmin_yap.py
```

#### 3. Kendi Projelerinizde Kullanım (Pipeline)
`src/pipeline.py` içerisindeki `musteri_kayip_tahmini` fonksiyonu bir uygulamaya, web sitesine veya farklı bir betiğe rahatça entegre edilecek şekilde tasarlanmıştır. İlgili dosyaya canlı bir müşteri verisi sözlük (dictionary) şeklinde iletildiğinde model tarafından oranlar geri döndürülür.

## 📊 Başarı Metrikleri

Projedeki güncel Lojistik Regresyon modeli sınıf dengesizliğini çözmek için **SMOTE** algoritması kullanarak başarılı şekilde optimize edilmiştir:
- **Duyarlılık (Recall - Sınıf 1):** %74 (Hedefimiz olan riskli müşterileri yüksek oranda tespit eder)
- **Doğruluk (Accuracy):** %76

## 🛠️ Kullanılan Teknolojiler
* **Python**, **Pandas**, **NumPy** (Veri İşleme ve Manipülasyonu)
* **Scikit-Learn**, **Imbalanced-learn (SMOTE)** (Makine Öğrenimi ve Modelleme)
* **Matplotlib**, **Seaborn** (Veri Görselleştirme)
* **Joblib** (Model İhracı / Yükleme)
