# Advanced Facial Recognition & Web Search Engine

🎯 Production-ready sistem yüz tanıma ve internet araması için.

## ✨ Özellikler

✅ **Yüz Tanıma**: Deep learning tabanlı yüz algılama ve tanıma  
✅ **Batch İşleme**: Paralel resim işleme  
✅ **Ters Görüntü Arama**: Bing + Google entegrasyonu  
✅ **Veritabanı**: Sonuçları sakla ve ara  
✅ **REST API**: Web servisi entegrasyonu  
✅ **Caching**: Performans optimizasyonu  
✅ **İstatistikler**: Detaylı arama ve işlem istatistikleri  
✅ **GPU Desteği**: CUDA ve TensorFlow entegrasyonu

## 📋 Gereksinimler

- Python 3.8+
- pip
- Face Recognition API Key (isteğe bağlı)
- Bing Search API Key (isteğe bağlı)
- Google Custom Search API Key (isteğe bağlı)

## 🚀 Kurulum

### 1. Depoyu Klonla

```bash
git clone https://github.com/ahtapotdyr-web/ahtapotdyr.git
cd ahtapotdyr
```

### 2. Virtual Environment Oluştur

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 4. Environment Değişkenlerini Ayarla

```bash
cp .env.example .env
# .env dosyasını edit edin ve API key'lerinizi ekleyin
```

### 5. Uygulamayı Çalıştır

```bash
python main.py
```

Uygulama http://localhost:5000 adresinde çalışacak.

## 💡 API Kullanımı

### 1. Resim Yükle

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/upload
```

**Yanıt:**
```json
{
  "success": true,
  "image_id": 1,
  "filename": "1234567890_photo.jpg",
  "face_count": 2,
  "statistics": {
    "total_faces": 2,
    "image_size": [1920, 1080],
    "faces": [...]
  }
}
```

### 2. Benzer Yüzleri Ara

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "image_id": 1,
    "face_index": 0,
    "tolerance": 0.6
  }' \
  http://localhost:5000/api/search/faces
```

### 3. Ters Görüntü Ara

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "image_id": 1,
    "query": "face detection",
    "count": 50
  }' \
  http://localhost:5000/api/search/reverse
```

### 4. Resimleri Listele

```bash
curl http://localhost:5000/api/images
```

### 5. İstatistikler

```bash
curl http://localhost:5000/api/stats
```

## 🔧 Konfigürasyon

Aşağıdaki değişkenleri `.env` dosyasında ayarlayabilirsin:

- `FLASK_ENV`: Ortam (development/production)
- `SECRET_KEY`: Flask gizli anahtarı
- `FACE_RECOGNITION_DISTANCE_THRESHOLD`: Yüz eşleştirme hassasiyeti (0.3-0.9)
- `USE_GPU`: GPU kullanımı (True/False)
- `MAX_SEARCH_RESULTS`: Maksimum arama sonucu sayısı
- `BING_API_KEY`: Bing Search API key'i
- `GOOGLE_API_KEY`: Google Custom Search API key'i

## 📊 Veritabanı

Sistem otomatik olarak SQLite veritabanı oluşturur. Aşağıdaki tablolar kullanılır:

- `images`: Yüklenen resimler ve yüz kodlamaları
- `search_history`: Arama geçmişi
- `search_results`: Arama sonuçları

## 🎯 Kullanım Örneği

```python
import requests

# Resim yükle
with open('photo.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    image_id = response.json()['image_id']

# Benzer yüzleri ara
response = requests.post('http://localhost:5000/api/search/faces', json={
    'image_id': image_id,
    'face_index': 0,
    'tolerance': 0.6
})

matches = response.json()['matches']
print(f"{len(matches)} benzer yüz bulundu")
```

## 🚀 Performans

- Yüz tanıma: ~1-2 saniye/resim
- Ters image arama: ~2-5 saniye
- Benzer yüz arama: ~0.5-1 saniye
- GPU ile 2-3x daha hızlı

## 🔒 Güvenlik

- API key'lerinizi `.env` dosyasında sakla
- Production ortamında güçlü secret key kullan
- CORS ayarlarını kısıtla
- Rate limiting ekle
- HTTPS kullan

## 📝 Lisans

MIT License - Ayrıntılar için LICENSE dosyasına bak.

## 👨‍💻 Geliştirici

**ahtapotdyr-web** - GitHub: [@ahtapotdyr-web](https://github.com/ahtapotdyr-web)

---

**Soru veya önerileriniz için bir issue açabilirsiniz!**
