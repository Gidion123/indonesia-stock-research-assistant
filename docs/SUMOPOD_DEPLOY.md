# Deploy RAG di VPS SumoPod

Target: Ubuntu 24.04, 2 vCPU, 4 GB RAM, 60 GB disk. Satu VPS menjalankan Streamlit, PostgreSQL/pgvector, dan pembersih unggahan sementara. CPU dipakai untuk embedding; jawaban menggunakan API DeepSeek. Kapasitas untuk banyak pengguna bersamaan belum diukur. SmartSplit Bill belum termasuk stack ini.

Status terakhir: pemilik proyek mengonfirmasi aplikasi lokal dan percakapan lanjutan POWR berhasil, dengan **377 passed, 4 skipped**. Image Docker terbaru berhasil dibangun secara lokal. VPS belum dinyatakan berhasil deploy sampai pemeriksaan di bawah selesai. Pengujian lokal tidak menggantikan pengujian di arsitektur CPU dan jaringan VPS.

Folder `data/knowledge_base/primary/` menjadi sumber indeks kurasi. Hasil yang diharapkan: **5 PDF, 117 halaman total, 105 halaman dipakai, 489 chunk, embedding 384 dimensi**. Database lokal tidak perlu disalin; indeks dibuat lagi dari PDF di VPS.

## 1. Login SSH pertama — dari Terminal Mac

Ganti `VPS_IP` dengan Public IP dari panel SumoPod, dan sesuaikan username jika berbeda:

```bash
ssh ubuntu@VPS_IP
```

Pada koneksi pertama, SSH meminta konfirmasi identitas server. Cocokkan fingerprint dengan hasil perintah berikut yang dijalankan melalui console VPS tepercaya dari panel penyedia:

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Setelah cocok, terima host key dan masukkan password VPS langsung di terminal jika diminta. Karakter password tidak tampil saat diketik. Jangan kirim password, private key, atau API key melalui chat. Jika koneksi timeout, periksa apakah VPS aktif dan port SSH di firewall SumoPod mengizinkan IP komputer Anda.

Agar deployment selanjutnya dapat dijalankan tanpa prompt password, pasang public key Mac pada `~/.ssh/authorized_keys` milik pengguna `ubuntu`. Jika belum punya key, buat dari terminal Mac kedua; jangan menimpa key yang sudah ada:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/sumopod_portfolio -C sumopod-portfolio
```

Salin **public key** saja ke VPS (perintah ini mungkin masih meminta password VPS):

```bash
cat ~/.ssh/sumopod_portfolio.pub | ssh ubuntu@VPS_IP \
  'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys'
ssh-add ~/.ssh/sumopod_portfolio
ssh -o BatchMode=yes ubuntu@VPS_IP 'whoami'
```

Hasil terakhir harus `ubuntu`. Private key tetap di Mac. `ssh-add` mungkin perlu diulang setelah komputer di-restart. Jika Anda sudah memakai key lain yang berfungsi, gunakan key itu.

## 2. Periksa server dan Docker — di VPS

```bash
uname -m
free -h
df -h /
docker --version
sudo docker compose version
```

Jika Docker/Compose belum tersedia, pasang **Docker Engine dan Compose plugin** melalui [repository resmi Docker untuk Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-apt-repository). Panduan tersebut mendukung Ubuntu 24.04. Periksa paket/container yang sudah ada sebelum mengganti instalasi Docker. Setelah pemasangan:

```bash
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

Build pertama memerlukan akses keluar ke repository paket, Docker Hub, PyTorch, dan PyPI; bootstrap mengunduh model dari Hugging Face. Jawaban memerlukan akses ke DeepSeek, dan pertanyaan harga memakai Yahoo Finance.

Untuk tahap ini, cukup izinkan port SSH di firewall penyedia. Compose tidak mempublikasikan PostgreSQL; Streamlit terikat ke loopback VPS. Jangan mengandalkan UFW saja untuk membatasi port container yang dipublikasikan, karena Docker mempunyai aturan jaringan sendiri ([catatan firewall Docker](https://docs.docker.com/engine/install/ubuntu/#firewall-limitations)).

## 3. Salin source terpilih — dari Terminal Mac

Jalankan dari folder proyek yang sudah Anda uji. Pastikan SSH login berhasil dahulu. Buat direktori tujuan:

```bash
ssh ubuntu@VPS_IP 'mkdir -p ~/rag-stock-customer-assistant'
```

Perintah berikut hanya mengirim file yang dibutuhkan, lima PDF primer, serta dokumentasi. `.env`, `.env.vps`, virtual environment, `.git`, dan database lokal tidak disalin. `-R` mempertahankan susunan direktori:

```bash
rsync -avR \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
  ./app.py ./Dockerfile ./.dockerignore ./requirements-vps.txt \
  ./compose.vps.yaml ./.env.vps.example ./README.md ./docs \
  ./src ./scripts ./.streamlit/config.toml ./data/knowledge_base/primary \
  ubuntu@VPS_IP:~/rag-stock-customer-assistant/
```

Perintah tidak memakai `--delete`. Untuk memperbarui proyek di kemudian hari, gunakan daftar source yang sama dan periksa jika ada file source lama yang memang perlu dihapus. `.dockerignore` juga membatasi build context ke source runtime dan PDF primer.

## 4. Buat kredensial — di VPS

```bash
cd ~/rag-stock-customer-assistant
umask 077
test -f .env.vps || cp .env.vps.example .env.vps
chmod 600 .env.vps
nano .env.vps
```

Isi `POSTGRES_PASSWORD` dengan nilai acak hexadecimal panjang dan `DEEPSEEK_API_KEY` dengan key Anda, langsung di editor VPS. Nilai hexadecimal aman disisipkan pada URL database. Untuk membuat password, Anda dapat menjalankan `openssl rand -hex 32` di terminal VPS sendiri, lalu menyalinnya ke editor. Jangan membagikan output password atau isi `.env.vps`.

File ini merupakan konfigurasi VPS; `.env` Mac tidak digunakan. Saat memakai Compose, gunakan `--env-file .env.vps` secara konsisten. Validasi dengan `config --quiet`, karena `config` tanpa opsi tersebut menampilkan konfigurasi yang telah berisi rahasia. Variabel dengan nama sama yang sudah di-export di shell dapat menimpa nilai file ([prioritas variabel Compose](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)).

Mengganti `POSTGRES_PASSWORD` pada file setelah volume PostgreSQL terbentuk tidak otomatis mengganti password di dalam database. Rotasi password perlu dilakukan pada keduanya.

## 5. Build dan buat indeks — di VPS

Dari direktori aplikasi:

```bash
sudo docker compose --env-file .env.vps -f compose.vps.yaml config --quiet
sudo docker compose --env-file .env.vps -f compose.vps.yaml build
sudo docker compose --env-file .env.vps -f compose.vps.yaml up -d db
sudo docker compose --env-file .env.vps -f compose.vps.yaml run --rm app python -m scripts.bootstrap_vps
sudo docker compose --env-file .env.vps -f compose.vps.yaml up -d app cleanup
sudo docker compose --env-file .env.vps -f compose.vps.yaml ps
```

Compose menunggu database sehat sebelum menjalankan bootstrap. Bootstrap mengunduh embedding ke volume cache, membuat tabel bila belum ada, lalu mengindeks PDF. Hasil pertama harus memuat **5 PDF, 105 halaman dipakai, 489 chunk**. Bila build/bootstrap gagal, selesaikan error dahulu sebelum menjalankan langkah berikutnya.

Bootstrap berikutnya melewati tabel yang sudah terisi; ini tidak memeriksa apakah isi PDF berubah. Model tersimpan di `rag_hf_cache`, database di `rag_postgres`. Nama volume sebenarnya memiliki prefix proyek Compose. Gunakan folder/nama proyek yang sama saat memperbarui deployment agar volume yang sama dipakai. Hindari `docker compose down -v` jika data ingin dipertahankan.

Periksa aplikasi dan indeks:

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
sudo docker compose --env-file .env.vps -f compose.vps.yaml exec -T db \
  psql -U rag -d rag_stock_assistant -c 'SELECT COUNT(*) AS chunks FROM stock_knowledge;'
sudo docker compose --env-file .env.vps -f compose.vps.yaml logs --tail=50 app cleanup
```

Endpoint health harus menjawab `ok`, dan query database menghasilkan `489`. Health check Streamlit hanya menandakan proses aplikasi hidup; tetap lakukan uji pertanyaan untuk memeriksa database dan API. Pembersih menjalankan penghapusan dokumen kedaluwarsa setiap jam; retrieval menolak dokumen yang kedaluwarsa meskipun belum dihapus.

## 6. Uji aplikasi VPS — dari Terminal Mac kedua

Aplikasi lokal Anda memakai port 8501. Gunakan port lokal **8502** untuk tunnel agar keduanya dapat berjalan bersamaan:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8502:127.0.0.1:8501 ubuntu@VPS_IP
```

Biarkan terminal ini terbuka, lalu buka [http://127.0.0.1:8502](http://127.0.0.1:8502). Ini adalah aplikasi **VPS** melalui SSH; `localhost:8501` tetap aplikasi Mac. Tunnel hanya memberi akses di komputer Anda, belum link publik.

Periksa:

1. Lima sumber primer tampil, termasuk `Riset Saham Data Center Indonesia.pdf`.
2. Dalam satu sesi, tanyakan prospek Cikarang Listrindo, kemudian `Apa target harga?` dan `Bagaimana valuasi?`; periksa sumber kutipannya.
3. Jika perlu, uji `Berapa harganya sekarang?` untuk memastikan akses Yahoo dari VPS. Harga dapat tertunda atau request terkena rate limit.
4. Buka tab/sesi kedua. Upload PDF kecil dengan teks unik pada sesi pertama; PDF dan teks privatnya harus tidak tersedia pada sesi kedua.
5. Periksa batas unggahan 5 MB, 30 halaman, dua PDF, serta batas pertanyaan. Uji ini mengonsumsi kuota dan pertanyaan aplikasi dapat memakai API berbayar.

Jika gagal, lihat log pada VPS. Tidak perlu mengirim API key atau isi `.env.vps` untuk diagnosis.

## 7. Link publik dan proyek kedua

Rancangan publik memakai domain dan reverse proxy HTTPS, misalnya Caddy. Setelah domain tersedia, arahkan DNS ke Public IP VPS, konfigurasi proxy untuk RAG, lalu buka port 80/443 dan uji HTTPS serta koneksi chat Streamlit. PostgreSQL dan port 8501 tetap privat. Proxy harus mendukung WebSocket ([panduan Docker Streamlit](https://docs.streamlit.io/deploy/tutorials/docker)).

Contoh susunan setelah kedua aplikasi selesai dideploy:

- `https://rag.nama-domain-anda` untuk RAG.
- `https://bill.nama-domain-anda` untuk SmartSplit Bill.

Nama di atas hanya contoh. Satu VPS dapat melayani dua alamat melalui reverse proxy; aplikasi kedua perlu konfigurasi dan uji penggunaan RAM tersendiri. File konfigurasi proxy/domain belum disertakan karena domain belum dipilih. Domain belum diperlukan untuk menjalankan langkah uji VPS melalui tunnel.

## 8. Backup dan pembaruan

Sebelum publikasi atau penggantian indeks, buat backup di VPS dan simpan salinannya di luar VPS dengan akses terbatas:

```bash
umask 077
sudo docker compose --env-file .env.vps -f compose.vps.yaml exec -T db \
  pg_dump -U rag -d rag_stock_assistant -Fc > "rag_stock_assistant-$(date +%Y%m%d-%H%M%S).dump"
```

Dump dapat memuat teks unggahan pengunjung yang belum dihapus. File `.dump` diabaikan Git; jangan masukkan backup atau `.env.vps` ke repository. Volume persisten membantu saat container diganti, tetapi tidak melindungi dari kehilangan VPS. Pemulihan backup belum diuji dalam deployment ini.

Untuk perubahan source, ulangi transfer terpilih dan build. Jika PDF atau aturan preprocessing berubah, lakukan pada waktu sepi:

```bash
sudo docker compose --env-file .env.vps -f compose.vps.yaml build
sudo docker compose --env-file .env.vps -f compose.vps.yaml stop app
sudo docker compose --env-file .env.vps -f compose.vps.yaml run --rm app python -m scripts.bootstrap_vps --rebuild
sudo docker compose --env-file .env.vps -f compose.vps.yaml up -d app cleanup
```

`--rebuild` mengganti indeks kurasi. Setelah berhasil, ulangi health check dan uji pertanyaan. Perubahan source saja biasanya cukup dengan `up -d --build app cleanup`, tanpa rebuild indeks. Model embedding/aturan chunking yang berubah memerlukan ingestion ulang.
