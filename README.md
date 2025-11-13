# Construction Site PPE Safety Monitor

Fast, end‑to‑end PPE compliance monitoring for construction sites. It ingests live cameras (YouTube/RTSP/USB), performs YOLOv8 detections, tracks people, evaluates PPE and zone rules, persists events, and serves a real‑time dashboard with analytics and alerts.

Highlights
- Live multi‑camera streaming via WebSocket (OpenCV + FFMPEG) with YouTube support (yt‑dlp).
- YOLOv8 inference (Ultralytics) with hot‑reload of custom weights.
- ByteTrack IDs, dwell‑time and geofencing rules, PPE compliance (helmet, vest, gloves, boots).
- JWT login (admin/operator), audit logs, configurable thresholds and class mapping.
- Alerts: webhook, email, SMS (Twilio) with rate‑limiting and escalation.
- Analytics, CSV/PDF exports, Prometheus metrics. Optional Docker/Compose with Postgres + Grafana.


Quick start (Windows PowerShell)
1) Create and activate a venv

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
# If PyTorch is missing, install CPU build first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

3) Run backend and frontend

```powershell
python app.py
python -m http.server 8001
```

- API: http://127.0.0.1:8000
- UI:  http://127.0.0.1:8001/main.html

Default login
- username: admin
- password: admin123  (change with `ADMIN_PASSWORD` env)


Live cameras (YouTube/RTSP/USB)
- In the “Live Camera Feeds (YouTube)” section, four tiles are pre‑filled with demo YouTube URLs. Press Start/Start All to begin server‑side processing.
- Any source supported by OpenCV/FFMPEG works: `0` (webcam), `rtsp://...`, `http(s)://video.mp4`, public YouTube URL.
- For YouTube, the backend resolves the direct stream using yt‑dlp and opens it with `cv2.CAP_FFMPEG` for reliability.


Key features
- WebSocket `/ws`: stream annotated JPEG frames per source.
- `/events`, `/analytics/summary`, `/reports/*`: historical insights and exports.
- Rule engine: `require_helmet`, `require_vest`, zone no‑go, dwell‑time.
- Model management: `/model`, `/model/reload`, hot thresholds in `/config`.
- Privacy: face blurring and data retention policy.


API cheatsheet (PowerShell)
1) Login
```powershell
$token = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/auth/login `
  -ContentType 'application/json' -Body (@{username='admin';password='admin123'} | ConvertTo-Json)).access_token
```

2) Reload weights
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/model/reload `
  -Headers @{ Authorization = \"Bearer $token\" } -ContentType 'application/json' `
  -Body (@{ weights = 'runs/ppe/exp_ppe/weights/best.pt' } | ConvertTo-Json)
```

3) Configure PPE classes, rules, thresholds (example mapping)
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/config `
  -Headers @{ Authorization = \"Bearer $token\" } -ContentType 'application/json' `
  -Body (@{
    ppe_classes = @{ person=7; helmet=2; vest=8; gloves=1; boots=0 };
    rules       = @{ require_helmet=$true; require_vest=$true };
    thresholds  = @{ conf=0.6 }
  } | ConvertTo-Json -Depth 5)
```

Endpoints (selection)
- POST `/predict` – image upload → annotated image + predictions.
- WS `/ws` – `{"source":"0"|"rtsp://..."|"https://youtube..."} → base64 JPEG frames`.
- GET `/events`, `/zones` – POST/DELETE (admin) to manage zones.
- GET/POST `/config` – get/update thresholds, class mapping, rules.
- GET `/model`, POST `/model/reload` – model status/load new weights.
- GET `/health` – liveness/readiness.
- GET `/analytics/summary`, `/reports/events.csv`, `/reports/summary.pdf`.


Train a custom PPE model (your dataset)
If your Roboflow export contains only a `train/` folder, set a temporary split (train/val/test → train) in `PPE.v1i.yolov8\data.yaml`:

```yaml
path: C:/Users/Kiruthik Kumar M/construction-site-safty/PPE.v1i.yolov8
train: C:/Users/Kiruthik Kumar M/construction-site-safty/PPE.v1i.yolov8/train/images
val:   C:/Users/Kiruthik Kumar M/construction-site-safty/PPE.v1i.yolov8/train/images
test:  C:/Users/Kiruthik Kumar M/construction-site-safty/PPE.v1i.yolov8/train/images
names: ['boots','gloves','hardhat','no_boots','no_gloves','no_hardhat','no_vest','person','vest']
```

Start training (outputs `runs/ppe/exp_ppe/weights/best.pt`):
```powershell
python train_ppe.py --data .\\PPE.v1i.yolov8\\data.yaml --model yolov8m.pt --epochs 60 --imgsz 640 --batch 16 --project runs/ppe --name exp_ppe
```

Load the trained weights and configure mapping/thresholds (see API cheatsheet). Planned mapping from the dataset above:
- person=7, helmet=2, vest=8, gloves=1, boots=0
- thresholds.conf=0.6 (tune for FP/FN trade‑off)

For production, create a proper `val/` and `test/` split to get unbiased metrics (mAP@50, precision/recall per class).


Docker (optional)
```bash
docker compose up --build
```
Services
- api: FastAPI (http://localhost:8000)
- frontend: static HTML (http://localhost:8001/main.html)
- postgres: DB (default credentials in compose)
- prometheus: http://localhost:9090
- grafana: http://localhost:3000


Operational notes
- DB: SQLite by default (`events.db`). Set `POSTGRES_DSN` for Postgres.
- Auth: JWT, roles admin/operator. Default admin password can be set via `ADMIN_PASSWORD` or changed after login.
- Alerts: `ALERT_WEBHOOK_URL` for webhooks; SMTP_* for email; Twilio env for SMS escalation.
- Metrics: Prometheus at `/metrics` and app counters for FPS, events, errors.
- Privacy: optional face blurring; retention job deletes older events on schedule.


Troubleshooting
- YouTube tile red dot: try another public link; ensure internet; backend logs should show “frame read ok”. Some videos are DASH‑only; yt‑dlp resolves progressive streams where available.
- “Only person detected”: load your trained PPE weights and set class mapping in `/config`. Adjust `thresholds.conf` (0.5–0.7) for your site.
- PowerShell API calls: always build JSON with `@{} | ConvertTo-Json`. Assign the token to `$token` as shown above.
- Prometheus “duplicated timeseries”: fixed by running `uvicorn.run(app, ...)` with the `app` object (avoid module double‑import).
- Bcrypt errors: we use `pbkdf2_sha256` for password hashing to avoid 72‑byte bcrypt limitations.
- Port already in use: stop other uvicorn/python processes or change ports.


Repository hygiene
- Large artifacts are ignored via `.gitignore` (weights `.pt`, `runs/`, `events.db`, videos, dataset images). Use Git LFS if you want to version weights:
```powershell
git lfs install
git lfs track \"*.pt\"
git add .gitattributes
git commit -m \"Track weights with Git LFS\"
git push
```


Security & licensing
- Ultralytics YOLOv8 is AGPL‑3.0; check license terms for your use case.
- Configure HTTPS, rotate JWT secrets, and restrict admin endpoints in production.


Demo script (3–5 minutes)
1) Start backend and UI; show `/health`. Login as admin.
2) Press “Start All” on YouTube tiles; point out live bounding boxes and violation list.
3) Draw a no‑go polygon; walk through a tracked person crossing → event appears.
4) Reload a trained PPE model via `/model/reload`, set class mapping via `/config`.
5) Open Analytics; export CSV or PDF; show Prometheus metrics page.

