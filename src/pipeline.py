import pandas as pd
import joblib

def musteri_kayip_tahmini(musteri_verisi, model_yolu, scaler_yolu):
    """
    Yeni müşteri verisini işler ve kayıp (churn) tahmini yapar.
    """
    # 1. Model ve ölçeklendiricinin disken okunması
    model = joblib.load(model_yolu)
    scaler = joblib.load(scaler_yolu)

    # 2. Sözlük formatındaki verinin DataFrame'e dönüştürülmesi
    df = pd.DataFrame([musteri_verisi])

    # 3. Özellik Mühendisliği işleminin uygulanması
    hizmet_sutunlari = [
        'OnlineSecurity_Yes', 'OnlineBackup_Yes',
        'DeviceProtection_Yes', 'TechSupport_Yes',
        'StreamingTV_Yes', 'StreamingMovies_Yes'
    ]
    
    # Eksik hizmet sütunu varsa hata vermemesi için kontrol
    for sutun in hizmet_sutunlari:
        if sutun not in df.columns:
            df[sutun] = 0

    df['Ek_Hizmet_Sayisi'] = df[hizmet_sutunlari].sum(axis=1)

    # 4. Sayısal değişkenlerin eğitim formatına göre ölçeklendirilmesi
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df[['tenure', 'MonthlyCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges']])

    # 5. Model üzerinden tahmin ve olasılık değerlerinin hesaplanması
    # Modele gönderilen sütun sırasının eğitim verisiyle aynı olduğundan emin olunmalıdır
    tahmin_sonucu = model.predict(df)[0]
    kayip_olasiligi = model.predict_proba(df)[0][1]

    return {
        "Tahmin": int(tahmin_sonucu), 
        "Kayıp_Olasılığı": round(float(kayip_olasiligi), 4)
    }