import pandas as pd
import joblib
import os

# 1. Dosya yollarının tanımlanması
# Kod scripts klasöründe çalışacağı için terminalden nerede çalıştırılırsa çalışsın mutlak yollar aranır.
model_yolu = r'C:\Users\Mustafa\Desktop\Customer_Churn_Prediction\models\lojistik_regresyon_modeli.pkl'
test_verisi_yolu = r'C:\Users\Mustafa\Desktop\Customer_Churn_Prediction\data\processed\islenmis_test_verisi.csv'

try:
    # 2. Modelin ve test verisinin sisteme yüklenmesi
    model = joblib.load(model_yolu)
    df_test = pd.read_csv(test_verisi_yolu)

    # 3. Test verisinden rastgele bir müşteri satırının çekilmesi
    rastgele_musteri = df_test.sample(n=1)

    # 4. Hedef değişkenin (Churn) modelden saklanması
    gercek_durum = rastgele_musteri['Churn'].values[0]
    tahmin_verisi = rastgele_musteri.drop(columns=['Churn'])

    # 5. Modelin tahmin ve olasılık hesaplaması yapması
    tahmin_sonucu = model.predict(tahmin_verisi)[0]
    tahmin_olasiligi = model.predict_proba(tahmin_verisi)[0][1]

    # 6. Sonuçların raporlanması
    print("--- Canlı Sistem Tahmin Simülasyonu ---")
    print(f"Müşterinin Gerçek Durumu : {gercek_durum} (1: Kayıp, 0: Mevcut)")
    print(f"Modelin Kararı           : {tahmin_sonucu}")
    print(f"Kayıp Gerçekleşme Riski  : % {round(tahmin_olasiligi * 100, 2)}")

except FileNotFoundError as e:
    print(f"Dosya okuma hatası: {e}\nLütfen dosya yollarını kontrol ediniz.")