# Deploy Guide — tbi-tp2.hafizmuh.site

Stack: **Ubuntu (fresh) + Docker + Docker Compose + Nginx + Let's Encrypt SSL**

Tidak perlu install Nginx atau Python di host — semua jalan di dalam container.

---

## Prasyarat

Sebelum mulai, pastikan:
- [ ] DNS record `tbi-tp2.hafizmuh.site` → A record ke IP VPS sudah dibuat
- [ ] Port **80** dan **443** terbuka di firewall VPS / security group

---

## Langkah 1 — Setup VPS

```bash
# SSH ke VPS
ssh user@<ip-vps>

# Update sistem
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo apt install -y docker-compose-plugin

# Logout dan login ulang agar group docker aktif
exit
ssh user@<ip-vps>

# Verifikasi
docker --version
docker compose version
```

---

## Langkah 2 — Clone repo ke VPS

```bash
git clone https://github.com/<username>/<repo>.git ~/tbi-tp2
cd ~/tbi-tp2
```

---

## Langkah 3 — Build index di dalam container

Index perlu dibangun sekali sebelum app bisa dijalankan.

```bash
# Build image dulu
docker compose build app

# Jalankan indexing di dalam container
docker compose run --rm app python search_cli.py index

# Build LSI model (direkomendasikan, untuk semantic search)
docker compose run --rm app python search_cli.py lsi build
```

Hasil index tersimpan di `./index/` di host (via volume mount), jadi tidak hilang saat container restart.

---

## Langkah 4 — SSL: Dapatkan sertifikat Let's Encrypt

Certbot butuh Nginx aktif di port 80 untuk verifikasi domain.
Gunakan konfigurasi HTTP-only sementara dulu.

```bash
# Buat direktori certbot
mkdir -p deploy/certbot/conf deploy/certbot/www

# Pakai config HTTP-only sementara
cp deploy/nginx/app.conf.http-only deploy/nginx/app.conf

# Jalankan app + nginx
docker compose up -d app nginx

# Tunggu sampai app healthy (cek dengan)
docker compose ps

# Minta sertifikat SSL
docker compose run --rm --profile certbot certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email <email-anda@domain.com> \
    --agree-tos \
    --no-eff-email \
    -d tbi-tp2.hafizmuh.site

# Verifikasi cert berhasil dibuat
ls deploy/certbot/conf/live/tbi-tp2.hafizmuh.site/
```

---

## Langkah 5 — Aktifkan HTTPS

```bash
# Ganti ke config HTTPS penuh
cp deploy/nginx/app.conf.https deploy/nginx/app.conf 2>/dev/null || \
    cp deploy/nginx/app.conf.http-only deploy/nginx/app.conf

# Edit manual: hapus server block HTTP-only dan uncomment HTTPS block
# Atau: gunakan file app.conf yang sudah berisi kedua server block

# Reload nginx
docker compose exec nginx nginx -s reload
```

Atau restart semua:
```bash
docker compose down
docker compose up -d
```

Buka browser: **https://tbi-tp2.hafizmuh.site** ✓

---

## Langkah 6 — Auto-renew SSL (cron)

Sertifikat Let's Encrypt berlaku 90 hari. Tambahkan cron job untuk renew otomatis:

```bash
# Buka crontab
crontab -e

# Tambahkan baris ini (renew setiap hari jam 3 pagi)
0 3 * * * cd ~/tbi-tp2 && docker compose run --rm --profile certbot certbot renew --quiet && docker compose exec nginx nginx -s reload
```

---

## Perintah Operasional

```bash
cd ~/tbi-tp2

# Lihat status semua container
docker compose ps

# Lihat log app secara live
docker compose logs -f app

# Lihat log nginx
docker compose logs -f nginx

# Restart app saja (tanpa downtime nginx)
docker compose restart app

# Stop semua
docker compose down

# Start ulang semua
docker compose up -d
```

---

## Update Aplikasi

```bash
cd ~/tbi-tp2

# Pull kode terbaru
git pull

# Rebuild image dan restart (zero-downtime: nginx tetap jalan)
docker compose up -d --build app

# Jika ada perubahan yang butuh re-index
docker compose run --rm app python search_cli.py index
docker compose run --rm app python search_cli.py lsi build
docker compose restart app
```

---

## Troubleshooting

**Container app tidak mau start / crash:**
```bash
docker compose logs app --tail 50
```

**Nginx 502 Bad Gateway:**
```bash
# Cek apakah app container healthy
docker compose ps
# Jika app belum healthy, tunggu ~30 detik lalu coba lagi
docker compose logs app --tail 20
```

**Certbot gagal — "Connection refused" atau "Timeout":**
```bash
# Pastikan port 80 terbuka
sudo ufw allow 80
sudo ufw allow 443
sudo ufw reload

# Pastikan DNS sudah propagate
curl -I http://tbi-tp2.hafizmuh.site
```

**Nginx SSL error — cert file tidak ditemukan:**
```bash
# Pastikan certbot sudah berhasil generate cert
ls deploy/certbot/conf/live/tbi-tp2.hafizmuh.site/
# Harus ada: cert.pem  chain.pem  fullchain.pem  privkey.pem
```

**Index tidak ditemukan setelah deploy:**
```bash
# Rebuild index di dalam container
docker compose run --rm app python search_cli.py index
docker compose restart app
```

---

## Struktur file deploy

```
docker-compose.yml              # Orchestrasi semua container
Dockerfile                      # Build image FastAPI app
.dockerignore                   # File yang dikecualikan dari image
DEPLOY.md                       # Panduan ini

deploy/
├── nginx/
│   ├── app.conf                # Nginx config aktif (HTTP + HTTPS)
│   └── app.conf.http-only      # Config sementara saat setup SSL
└── certbot/
    ├── conf/                   # SSL certs (auto-generated, di-gitignore)
    └── www/                    # ACME challenge webroot
```