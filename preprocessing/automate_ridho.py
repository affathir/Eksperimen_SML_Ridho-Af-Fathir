import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import dump

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing...")
    
    # 1. Membaca data mentah
    raw_data_path = 'padi_raw/Data_Tanaman_Padi_Sumatera_version_1.csv'
    df = pd.read_csv(raw_data_path)
    
    # 2. Proses Pembersihan dan Transformasi
    df = df.dropna()
    
    label_encoder = LabelEncoder()
    df['Provinsi'] = label_encoder.fit_transform(df['Provinsi'])
    
    scaler = MinMaxScaler()
    kolom_numerik = ['Tahun', 'Produksi', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
    df[kolom_numerik] = scaler.fit_transform(df[kolom_numerik])
    
    # 3. Menyiapkan folder penyimpanan otomatis
    output_dir = 'padi_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Menyimpan dataset bersih dan artefak (scaler & encoder)
    df.to_csv(os.path.join(output_dir, 'padi_clean.csv'), index=False)
    dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    
    print("Selesai! Data bersih dan artefak telah disimpan di folder 'padi_preprocessing'.")

if __name__ == "__main__":
    run_preprocessing()