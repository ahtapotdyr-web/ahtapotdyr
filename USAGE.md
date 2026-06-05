# Kullanım Kılavuzu

## Başlangıç

### 1. Adım: Kurulum

```bash
git clone https://github.com/ahtapotdyr-web/ahtapotdyr.git
cd ahtapotdyr
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

### 2. Adım: Konfigürasyon

```bash
cp .env.example .env
# .env dosyasını edit edin ve kişisel bilgilerinizi ekleyin
```

### 3. Adım: Çalıştırma

```bash
python main.py
```

Sunucu http://localhost:5000 adresinde çalışacak.

## Python Scriptiyle Kullanım

### Basit Örnek

```python
import requests
import json

BASE_URL = 'http://localhost:5000/api'

# 1. Resim yükle
with open('my_photo.jpg', 'rb') as f:
    response = requests.post(f'{BASE_URL}/upload', files={'file': f})
    result = response.json()
    image_id = result['image_id']
    
    print(f"Resim yüklendi: ID={image_id}")
    print(f"Bulunan yüz sayısı: {result['face_count']}")

# 2. Benzer yüzleri ara
response = requests.post(f'{BASE_URL}/search/faces', json={
    'image_id': image_id,
    'face_index': 0,
    'tolerance': 0.6
})

matches = response.json()['matches']
print(f"\\nBenzer yüzler: {len(matches)} bulundu")
for match in matches:
    print(f"  - {match['filename']}: {match['similarity']:.2%} benzerlik")

# 3. Web'de ara
response = requests.post(f'{BASE_URL}/search/reverse', json={
    'image_id': image_id,
    'query': 'yüz tanıma',
    'count': 50
})

results = response.json()['results']
print(f"\\nInternet arama sonuçları: {len(results)} bulundu")
for result in results[:5]:
    print(f"  - {result['title']} ({result['source']})")
    print(f"    URL: {result['url']}")
```

### Gelişmiş Örnek

```python
import requests
import json
from pathlib import Path

class FacialRecognitionClient:
    def __init__(self, base_url='http://localhost:5000/api'):
        self.base_url = base_url
    
    def upload_image(self, image_path):
        """Resim yükle"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f'{self.base_url}/upload',
                files={'file': f}
            )
        return response.json()
    
    def search_similar_faces(self, image_id, face_index=0, tolerance=0.6):
        """Benzer yüzleri ara"""
        response = requests.post(
            f'{self.base_url}/search/faces',
            json={
                'image_id': image_id,
                'face_index': face_index,
                'tolerance': tolerance
            }
        )
        return response.json()
    
    def search_reverse(self, image_id, query='', count=50):
        """Ters görüntü ara"""
        response = requests.post(
            f'{self.base_url}/search/reverse',
            json={
                'image_id': image_id,
                'query': query,
                'count': count
            }
        )
        return response.json()
    
    def list_images(self):
        """Tüm resimleri listele"""
        response = requests.get(f'{self.base_url}/images')
        return response.json()
    
    def get_stats(self):
        """İstatistikleri al"""
        response = requests.get(f'{self.base_url}/stats')
        return response.json()

# Kullanım
client = FacialRecognitionClient()

# Klasördeki tüm resimleri işle
image_dir = Path('my_images')
for image_path in image_dir.glob('*.jpg'):
    print(f"\\nİşleniyor: {image_path.name}")
    
    # Yükle
    upload_result = client.upload_image(image_path)
    image_id = upload_result['image_id']
    print(f"  ✓ Yüklendi (ID: {image_id})")
    print(f"  ✓ Yüz sayısı: {upload_result['face_count']}")
    
    # Benzer yüzleri ara
    face_result = client.search_similar_faces(image_id)
    print(f"  ✓ Benzer yüz: {len(face_result['matches'])} bulundu")
    
    # Web'de ara
    web_result = client.search_reverse(image_id)
    print(f"  ✓ Web sonuçları: {web_result['count']} bulundu")

# İstatistikleri göster
stats = client.get_stats()['statistics']
print(f"\\n=== İstatistikler ===")
print(f"Toplam resim: {stats['total_images']}")
print(f"Toplam yüz: {stats['total_faces']}")
print(f"Toplam arama: {stats['total_searches']}")
print(f"Toplam sonuç: {stats['total_results']}")
print(f"Ortalama arama süresi: {stats['avg_search_duration']:.2f}s")
```

## cURL Örnekleri

### Resim Yükle

```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:5000/api/upload
```

### Benzer Yüzleri Ara

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": 1,
    "face_index": 0,
    "tolerance": 0.6
  }' \
  http://localhost:5000/api/search/faces
```

### Web'de Ara

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": 1,
    "query": "face detection",
    "count": 50
  }' \
  http://localhost:5000/api/search/reverse
```

### Tüm Resimleri Listele

```bash
curl http://localhost:5000/api/images
```

### İstatistikleri Al

```bash
curl http://localhost:5000/api/stats
```

## Hata Yönetimi

```python
import requests

try:
    response = requests.post(f'{BASE_URL}/search/faces', json={
        'image_id': 999,
        'face_index': 0
    })
    
    if response.status_code == 404:
        print("Resim bulunamadı")
    elif response.status_code == 400:
        print("Geçersiz istek")
    elif response.status_code == 500:
        print("Server hatası")
    else:
        data = response.json()
        if data.get('success'):
            print("Başarılı!")
        else:
            print(f"Hata: {data.get('error')}")
            
except requests.exceptions.ConnectionError:
    print("Sunucuya bağlanılamıyor")
except Exception as e:
    print(f"Hata: {e}")
```

## İpuçları ve Tricks

### 1. Performansı Artır

```python
# Batch işleme yapabilirsin
image_dir = Path('images')
for image_file in image_dir.glob('*.jpg'):
    client.upload_image(image_file)
```

### 2. Sonuçları Filtrele

```python
result = client.search_similar_faces(image_id)
# Sadece yüksek benzerlik olanları al
high_matches = [m for m in result['matches'] if m['similarity'] > 0.9]
```

### 3. Hassasiyeti Ayarla

```python
# Daha katı: 0.3 (çok az eşleşme)
# Orta: 0.6 (dengeleme)
# Gevşek: 0.9 (çok fazla eşleşme)
result = client.search_similar_faces(image_id, tolerance=0.3)
```

---

Daha detaylı bilgi için README.md dosyasına bakın!
