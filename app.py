import io
import os
import time
import csv
import base64
import asyncio
import json
import sqlite3
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import requests
from jose import JWTError, jwt
from passlib.hash import pbkdf2_sha256
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

# Allow the frontend (served locally) to call this API. Adjust origins if you serve frontend on another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------
# Global state
# ------------------
model = None
db = None
db_driver = "sqlite"
db_path = "events.db"

# Simple per-source tracking state
track_states: Dict[str, Dict[str, object]] = {}
byte_trackers: Dict[str, object] = {}
dwell_state: Dict[str, Dict[int, Dict[int, float]]] = {}

# Config with env fallbacks
CONFIG = {
    "api_key": os.environ.get("API_KEY", "devkey"),
    "alert_webhook": os.environ.get("ALERT_WEBHOOK_URL", ""),
    "smtp": {
        "host": os.environ.get("SMTP_HOST", ""),
        "port": int(os.environ.get("SMTP_PORT", "0") or "0"),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASS", ""),
        "to": os.environ.get("ALERT_EMAIL_TO", ""),
    },
    # Default class mapping: COCO 'person' is 0; PPE classes require a PPE-trained model
    "ppe_classes": {"person": 0, "helmet": None, "vest": None},
    "rules": {
        "require_helmet": False,
        "require_vest": False,
        "dwell_seconds": 0
    },
    "privacy": {
        "face_blur": False,
        "retention_days": 30
    },
    "alerts": {
        "rate_seconds": 60,
        "escalate_after": 5
    },
    "thresholds": {
        "conf": 0.5
    },
    "jwt": {
        "secret": os.environ.get("JWT_SECRET", "change-me"),
        "algorithm": "HS256",
        "expiry_seconds": 3600
    },
    "twilio": {
        "sid": os.environ.get("TWILIO_ACCOUNT_SID", ""),
        "token": os.environ.get("TWILIO_AUTH_TOKEN", ""),
        "from": os.environ.get("TWILIO_FROM", ""),
        "to": os.environ.get("TWILIO_TO", ""),
        "escalate_to": os.environ.get("TWILIO_ESCALATE_TO", "")
    },
    "postgres_dsn": os.environ.get("POSTGRES_DSN", ""),
}


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if x_api_key != CONFIG["api_key"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def db_connect() -> sqlite3.Connection:
    global db_driver
    if CONFIG["postgres_dsn"]:
        try:
            import psycopg2  # type: ignore
            conn = psycopg2.connect(CONFIG["postgres_dsn"])
            db_driver = "postgres"
            return conn  # type: ignore
        except Exception as e:
            print("Failed to connect to Postgres, falling back to SQLite:", e)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    db_driver = "sqlite"
    return conn


def sql_norm(sql: str) -> str:
    if db_driver == "postgres":
        return sql.replace("?", "%s")
    return sql


def db_execute(sql: str, params: Tuple = ()) -> None:
    if db_driver == "sqlite":
        db.execute(sql, params)
    else:
        cur = db.cursor()
        cur.execute(sql_norm(sql), params)
        cur.close()


def db_fetchall(sql: str, params: Tuple = ()) -> list:
    if db_driver == "sqlite":
        return db.execute(sql, params).fetchall()
    cur = db.cursor()
    cur.execute(sql_norm(sql), params)
    rows = cur.fetchall()
    cur.close()
    return rows


def db_commit() -> None:
    db.commit()


def db_init(conn: sqlite3.Connection) -> None:
    # events
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id SERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            source TEXT,
            event_type TEXT,
            class_id INTEGER,
            conf DOUBLE PRECISION,
            x1 DOUBLE PRECISION, y1 DOUBLE PRECISION, x2 DOUBLE PRECISION, y2 DOUBLE PRECISION,
            track_id INTEGER
        )
        """
    )
    # zones
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS zones (
            id SERIAL PRIMARY KEY,
            source TEXT,
            name TEXT,
            type TEXT,
            polygon TEXT
        )
        """
    )
    # config (reserved)
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    # users
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT
        )
        """
    )
    # audit logs
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            username TEXT,
            action TEXT,
            detail TEXT
        )
        """
    )
    db_commit()


def point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    # Ray casting algorithm
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cond = ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1)
        if cond:
            inside = not inside
    return inside


def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-9)


def update_tracks_for_source(source: str, boxes: List[Tuple[float, float, float, float]]) -> List[int]:
    """Very simple IOU-based tracker per source; returns track ids for each box in order."""
    state = track_states.setdefault(source, {"next_id": 1, "tracks": []})
    tracks = state["tracks"]  # list of dict: {"id", "box", "last"}
    now = time.time()

    # Match existing tracks to new boxes
    assigned = [-1] * len(boxes)
    used_tracks = set()
    for ti, tr in enumerate(tracks):
        best_j = -1
        best_iou = 0.0
        for j, b in enumerate(boxes):
            if assigned[j] != -1:
                continue
            i = iou(tr["box"], b)
            if i > best_iou:
                best_iou = i
                best_j = j
        if best_j != -1 and best_iou >= 0.3:
            assigned[best_j] = tr["id"]
            tr["box"] = boxes[best_j]
            tr["last"] = now
            used_tracks.add(tr["id"])

    # Create new tracks for unmatched boxes
    for j, b in enumerate(boxes):
        if assigned[j] == -1:
            tid = state["next_id"]
            state["next_id"] += 1
            tracks.append({"id": tid, "box": b, "last": now})
            assigned[j] = tid

    # Remove stale tracks
    tracks[:] = [t for t in tracks if now - t["last"] <= 2.0]

    return assigned


def load_model_once():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("Model loaded: yolov8n.pt")
    except Exception as e:
        print("Failed to load ultralytics YOLO model:", e)
        model = None


def load_model_from(weights_path: str) -> bool:
    """Load YOLO model from given weights path. Returns True on success."""
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        print(f"Model loaded: {weights_path}")
        return True
    except Exception as e:
        print("Failed to load model:", e)
        return False


def blur_faces_bgr(img_bgr: np.ndarray) -> np.ndarray:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        out = img_bgr.copy()
        for (x, y, w, h) in faces:
            roi = out[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi = cv2.GaussianBlur(roi, (31, 31), 0)
            out[y:y+h, x:x+w] = roi
        return out
    except Exception:
        return img_bgr


def new_jwt(username: str, role: str) -> str:
    payload = {"sub": username, "role": role, "exp": int(time.time()) + CONFIG["jwt"]["expiry_seconds"]}
    return jwt.encode(payload, CONFIG["jwt"]["secret"], algorithm=CONFIG["jwt"]["algorithm"])


def decode_jwt(token: str) -> Dict[str, str]:
    return jwt.decode(token, CONFIG["jwt"]["secret"], algorithms=[CONFIG["jwt"]["algorithm"]])


def require_role(required: str):
    def dep(authorization: Optional[str] = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing token")
        token = authorization.split(" ", 1)[1]
        try:
            claim = decode_jwt(token)
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        role = claim.get("role", "operator")
        if required == "operator":
            return claim
        if required == "admin" and role == "admin":
            return claim
        raise HTTPException(status_code=403, detail="Forbidden")
    return dep


def audit(username: str, action: str, detail: str = "") -> None:
    try:
        db_execute("INSERT INTO audit_logs (ts, username, action, detail) VALUES (?,?,?,?)", (time.time(), username, action, detail))
        db_commit()
    except Exception:
        pass


@app.on_event("startup")
def startup():
    # Load model
    load_model_once()
    # Init DB
    global db
    db = db_connect()
    db_init(db)
    # Create default admin if none
    rows = db_fetchall("SELECT COUNT(*) FROM users")
    if rows and rows[0][0] == 0:
        default_admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
        db_execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)", ("admin", pbkdf2_sha256.hash(default_admin_pass), "admin"))
        db_commit()
        print("Created default admin user 'admin'")
    # Start retention task
    asyncio.create_task(retention_task())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an uploaded image, runs YOLO inference and returns a JSON with a base64 PNG of the annotated image and the raw predictions."""
    if model is None:
        return JSONResponse({"error": "Model not loaded on server."}, status_code=500)

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"Unable to read image: {e}"}, status_code=400)

    arr = np.array(image)

    # Run inference. Pass the numpy array (H,W,3)
    results = model(arr)
    if len(results) == 0:
        return {"image": None, "predictions": []}

    r = results[0]

    # Annotated image as numpy array
    try:
        annotated = r.plot()  # ultralytics result.plot()
    except Exception:
        # fallback: return original image
        annotated = arr

    # Privacy: optional face blur (RGB->BGR for OpenCV)
    if CONFIG["privacy"].get("face_blur"):
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        bgr = blur_faces_bgr(bgr)
        annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Convert annotated (numpy) to PNG bytes
    if annotated.dtype != np.uint8:
        annotated = (annotated * 255).astype(np.uint8)

    annotated_pil = Image.fromarray(annotated)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Extract simple box predictions if available
    preds = []
    try:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)
            # xyxy, conf, cls may be tensors; convert to lists
            if xyxy is not None:
                xyxy_list = xyxy.cpu().numpy().tolist()
            else:
                xyxy_list = []
            if conf is not None:
                conf_list = conf.cpu().numpy().tolist()
            else:
                conf_list = []
            if cls is not None:
                cls_list = cls.cpu().numpy().tolist()
            else:
                cls_list = []

            threshold = float(CONFIG["thresholds"].get("conf", 0.5) or 0.5)
            for i, box in enumerate(xyxy_list):
                c = int(cls_list[i]) if i < len(cls_list) else None
                cf = float(conf_list[i]) if i < len(conf_list) else None
                if cf is not None and cf < threshold:
                    continue
                preds.append({"xyxy": box, "class": c, "conf": cf})
    except Exception:
        preds = []

    return {"image": img_b64, "predictions": preds}


@app.websocket("/ws")
async def websocket_infer(websocket: WebSocket):
    """Streams live annotated frames over WebSocket. Client sends a JSON with {'source': 0 | '<rtsp/url>'} after connect."""
    await websocket.accept()
    if model is None:
        await websocket.close(code=1011)
        return

    cap = None
    source_label = "0"
    last_ping = time.time()
    try:
        # Receive initial source message (optional)
        try:
            init_msg = await websocket.receive_text()
            init = json.loads(init_msg) if init_msg else {}
        except Exception:
            init = {}

        source = init.get("source", 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        source_label = str(source)

        # Resolve YouTube URLs to direct streams if needed
        def resolve_source(src):
            try:
                if isinstance(src, str) and any(d in src for d in ("youtube.com", "youtu.be", "m.youtube.com")):
                    import yt_dlp  # type: ignore
                    ydl_opts = {
                        "quiet": True,
                        "nocheckcertificate": True,
                        "noplaylist": True,
                        # Prefer progressive (video+audio) HTTPS streams OpenCV can read via FFMPEG
                        "format": "bestvideo[ext=mp4][protocol^=https][vcodec!*=av01]+bestaudio[ext=m4a]/best[ext=mp4][protocol^=https]/best[protocol^=https]/best",
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(src, download=False)
                        if "entries" in info:
                            info = info["entries"][0]
                        # Try direct url first (progressive)
                        url = info.get("url")
                        if url:
                            return url
                        # Otherwise select a progressive format manually
                        fmts = info.get("formats") or []
                        candidates = []
                        for f in fmts:
                            if f.get("vcodec") != "none" and f.get("acodec") != "none":
                                if str(f.get("protocol", "")).startswith("https"):
                                    candidates.append(f)
                        if not candidates:
                            candidates = [f for f in fmts if f.get("vcodec") != "none" and f.get("acodec") != "none"]
                        if candidates:
                            # Pick highest bitrate
                            candidates.sort(key=lambda f: (f.get("tbr") or 0, f.get("height") or 0), reverse=True)
                            return candidates[0].get("url") or src
                        return src
            except Exception:
                pass
            return src

        src_resolved = resolve_source(source)
        try:
            cap = cv2.VideoCapture(src_resolved, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(src_resolved)
        if not cap or not cap.isOpened():
            # Send one-time error and close
            await websocket.send_text("")
            await websocket.close(code=1011)
            return

        while True:
            # Basic keepalive: if no activity, send ping as empty frame
            if time.time() - last_ping > 10:
                try:
                    await websocket.send_text("")
                except Exception:
                    break
                last_ping = time.time()

            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.1)
                continue

            # Run inference and annotate
            try:
                results = model(frame)
                r = results[0]
                annotated = r.plot()

                # Extract predictions
                preds_boxes = []
                preds_conf = []
                preds_cls = []
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    xyxy = getattr(boxes, "xyxy", None)
                    conf = getattr(boxes, "conf", None)
                    cls = getattr(boxes, "cls", None)
                    if xyxy is not None:
                        preds_boxes = xyxy.cpu().numpy().tolist()
                    if conf is not None:
                        preds_conf = conf.cpu().numpy().tolist()
                    if cls is not None:
                        preds_cls = cls.cpu().numpy().tolist()

                # Tracking IDs (very simple)
                track_ids = update_tracks_for_source(source_label, [tuple(map(float, b)) for b in preds_boxes])

                # Evaluate rules and persist events
                h, w = annotated.shape[:2]
                evaluate_and_persist_events(
                    db,
                    source_label,
                    preds_boxes,
                    preds_cls,
                    preds_conf,
                    track_ids,
                    frame_w=w,
                    frame_h=h,
                )
                # Privacy: optional face blur on annotated frame
                if CONFIG["privacy"].get("face_blur"):
                    annotated = blur_faces_bgr(annotated)
            except Exception:
                annotated = frame

            ok2, buf = cv2.imencode(".jpg", annotated)
            if not ok2:
                await asyncio.sleep(0.01)
                continue

            img_b64 = base64.b64encode(buf).decode("utf-8")
            await websocket.send_text(img_b64)

            # ~30 FPS target cap
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        pass
    except Exception:
        # Best effort close on unexpected errors
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


# ------------------
# Events, zones, rules, alerts
# ------------------
class Zone(BaseModel):
    source: str
    name: str
    type: str = "no_go"  # "no_go" or "info"
    polygon: List[Tuple[float, float]]  # normalized [0..1] points


class ConfigUpdate(BaseModel):
    ppe_classes: Optional[Dict[str, Optional[int]]] = None
    rules: Optional[Dict[str, bool]] = None
    alert_webhook: Optional[str] = None
    smtp: Optional[Dict[str, str]] = None
    api_key: Optional[str] = None
    privacy: Optional[Dict[str, object]] = None
    alerts: Optional[Dict[str, object]] = None
    jwt: Optional[Dict[str, object]] = None
    twilio: Optional[Dict[str, str]] = None


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "operator"


class LoginRequest(BaseModel):
    username: str
    password: str


def evaluate_and_persist_events(
    conn: sqlite3.Connection,
    source: str,
    boxes: List[List[float]],
    classes: List[float],
    confs: List[float],
    track_ids: List[int],
    frame_w: int,
    frame_h: int,
) -> None:
    """Evaluate simple rules and persist any resulting events."""
    ts = time.time()
    conf_thresh = float(CONFIG["thresholds"].get("conf", 0.5) or 0.5)

    # Load zones for source
    zones = db_fetchall("SELECT id, type, polygon FROM zones WHERE source = ?", (source,))
    parsed_zones = []
    for zid, ztype, poly_json in zones:
        try:
            poly = json.loads(poly_json)
            parsed_zones.append((zid, ztype, poly))
        except Exception:
            continue

    # For each detection, log a generic detection and check zones
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box)
        class_id = int(classes[i]) if i < len(classes) else -1
        conf = float(confs[i]) if i < len(confs) else 0.0
        tid = int(track_ids[i]) if i < len(track_ids) else None

        # Generic detection event (could be filtered by class/conf)
        if conf >= conf_thresh:
            db_execute(
                "INSERT INTO events (ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, source, "detection", class_id, conf, x1, y1, x2, y2, tid),
            )
            EVENTS_COUNTER.labels(event_type="detection").inc()

        # Zone rule: no_go violation if center is inside
        if parsed_zones:
            cx = (x1 + x2) / 2.0 / max(1.0, frame_w)
            cy = (y1 + y2) / 2.0 / max(1.0, frame_h)
            for zid, ztype, poly in parsed_zones:
                if ztype == "no_go" and point_in_polygon(cx, cy, [(float(px), float(py)) for px, py in poly]):
                    db_execute(
                        "INSERT INTO events (ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (ts, source, "zone_violation", class_id, conf, x1, y1, x2, y2, tid),
                    )
                    EVENTS_COUNTER.labels(event_type="zone_violation").inc()
                    trigger_alert({"type": "zone_violation", "source": source, "track_id": tid, "class_id": class_id, "conf": conf})

        # Dwell-time rule (if enabled)
        dwell_sec = int(CONFIG["rules"].get("dwell_seconds", 0) or 0)
        if dwell_sec > 0 and parsed_zones and tid is not None:
            state_src = dwell_state.setdefault(source, {})
            state_tid = state_src.setdefault(tid, {})
            cx = (x1 + x2) / 2.0 / max(1.0, frame_w)
            cy = (y1 + y2) / 2.0 / max(1.0, frame_h)
            for zid, ztype, poly in parsed_zones:
                if ztype != "no_go":
                    continue
                inside = point_in_polygon(cx, cy, [(float(px), float(py)) for px, py in poly])
                if inside:
                    first = state_tid.get(zid, ts)
                    state_tid[zid] = first
                    if ts - first >= dwell_sec:
                        db_execute(
                            "INSERT INTO events (ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (ts, source, "dwell_violation", class_id, conf, x1, y1, x2, y2, tid),
                        )
                        EVENTS_COUNTER.labels(event_type="dwell_violation").inc()
                        trigger_alert({"type": "dwell_violation", "source": source, "track_id": tid, "class_id": class_id, "conf": conf})
                        state_tid[zid] = ts + 1e9  # prevent repeated triggers
                else:
                    if zid in state_tid:
                        state_tid.pop(zid, None)

    # PPE rules: require helmet/vest if configured and class ids are known
    try:
        require_helmet = bool(CONFIG["rules"].get("require_helmet"))
        require_vest = bool(CONFIG["rules"].get("require_vest"))
        cls_map = CONFIG.get("ppe_classes", {})
        person_id = cls_map.get("person")
        helmet_id = cls_map.get("helmet")
        vest_id = cls_map.get("vest")
        if (require_helmet and helmet_id is not None) or (require_vest and vest_id is not None):
            person_idxs = [i for i, c in enumerate(classes) if int(c) == int(person_id)]
            helmet_boxes = [boxes[i] for i, c in enumerate(classes) if helmet_id is not None and int(c) == int(helmet_id) and float(confs[i]) >= 0.3]
            vest_boxes = [boxes[i] for i, c in enumerate(classes) if vest_id is not None and int(c) == int(vest_id) and float(confs[i]) >= 0.3]
            for pi in person_idxs:
                px1, py1, px2, py2 = map(float, boxes[pi])
                ptid = int(track_ids[pi]) if pi < len(track_ids) else None
                pbox = (px1, py1, px2, py2)
                if require_helmet and helmet_id is not None:
                    has_helmet = any(iou(pbox, tuple(map(float, hb))) >= 0.05 for hb in helmet_boxes)
                    if not has_helmet:
                        db_execute(
                            "INSERT INTO events (ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (time.time(), source, "ppe_violation_helmet", int(person_id), 1.0, px1, py1, px2, py2, ptid),
                        )
                        EVENTS_COUNTER.labels(event_type="ppe_violation_helmet").inc()
                        trigger_alert({"type": "ppe_violation_helmet", "source": source, "track_id": ptid, "class_id": int(person_id), "conf": 1.0})
                if require_vest and vest_id is not None:
                    has_vest = any(iou(pbox, tuple(map(float, vb))) >= 0.05 for vb in vest_boxes)
                    if not has_vest:
                        db_execute(
                            "INSERT INTO events (ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (time.time(), source, "ppe_violation_vest", int(person_id), 1.0, px1, py1, px2, py2, ptid),
                        )
                        EVENTS_COUNTER.labels(event_type="ppe_violation_vest").inc()
                        trigger_alert({"type": "ppe_violation_vest", "source": source, "track_id": ptid, "class_id": int(person_id), "conf": 1.0})
    except Exception:
        pass

    db_commit()


alert_last_sent: Dict[str, float] = {}
alert_count: Dict[str, int] = {}

def trigger_alert(event: Dict[str, object]) -> None:
    # Webhook
    if CONFIG.get("alert_webhook"):
        try:
            requests.post(CONFIG["alert_webhook"], json=event, timeout=2)
        except Exception:
            pass
    # Email (optional; SMTP must be configured)
    smtp_cfg = CONFIG.get("smtp", {})
    if smtp_cfg.get("host") and smtp_cfg.get("to"):
        try:
            import smtplib
            from email.mime.text import MIMEText

            body = json.dumps(event)
            msg = MIMEText(body)
            msg["Subject"] = f"Safety Alert: {event.get('type')}"
            msg["From"] = smtp_cfg.get("user") or "alerts@example.com"
            msg["To"] = smtp_cfg["to"]

            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg.get("port", 25), timeout=3) as server:
                if smtp_cfg.get("user") and smtp_cfg.get("password"):
                    server.starttls()
                    server.login(smtp_cfg["user"], smtp_cfg["password"])
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())
        except Exception:
            pass

    # SMS via Twilio (rate-limited with escalation)
    now = time.time()
    etype = str(event.get("type", "event"))
    last = alert_last_sent.get(etype, 0)
    if now - last >= int(CONFIG["alerts"].get("rate_seconds", 60) or 60):
        alert_last_sent[etype] = now
        alert_count[etype] = alert_count.get(etype, 0) + 1
        if CONFIG["twilio"]["sid"] and CONFIG["twilio"]["token"] and CONFIG["twilio"]["from"] and CONFIG["twilio"]["to"]:
            try:
                from twilio.rest import Client  # type: ignore
                client = Client(CONFIG["twilio"]["sid"], CONFIG["twilio"]["token"])
                body = f"Safety Alert: {etype} on {event.get('source')} (track {event.get('track_id')})"
                client.messages.create(body=body, from_=CONFIG["twilio"]["from"], to=CONFIG["twilio"]["to"])
                # Escalation
                if alert_count[etype] >= int(CONFIG["alerts"].get("escalate_after", 5) or 5) and CONFIG["twilio"]["escalate_to"]:
                    client.messages.create(body="[ESCALATION] " + body, from_=CONFIG["twilio"]["from"], to=CONFIG["twilio"]["escalate_to"])
                    alert_count[etype] = 0
            except Exception:
                pass


@app.get("/events")
def list_events(limit: int = 50):
    rows = db_fetchall("SELECT id, ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id FROM events ORDER BY id DESC LIMIT ?", (limit,))
    items = []
    for r in rows:
        items.append(
            {
                "id": r[0],
                "ts": r[1],
                "source": r[2],
                "type": r[3],
                "class_id": r[4],
                "conf": r[5],
                "xyxy": [r[6], r[7], r[8], r[9]],
                "track_id": r[10],
            }
        )
    return {"events": items}


@app.get("/zones")
def get_zones():
    rows = db_fetchall("SELECT id, source, name, type, polygon FROM zones")
    items = []
    for r in rows:
        try:
            poly = json.loads(r[4])
        except Exception:
            poly = []
        items.append({"id": r[0], "source": r[1], "name": r[2], "type": r[3], "polygon": poly})
    return {"zones": items}


@app.post("/zones")
def create_zone(zone: Zone, claim=Depends(require_role("admin"))):
    db_execute(
        "INSERT INTO zones (source, name, type, polygon) VALUES (?,?,?,?)",
        (zone.source, zone.name, zone.type, json.dumps(zone.polygon)),
    )
    db_commit()
    audit(claim["sub"], "create_zone", f"{zone.source}:{zone.name}")
    return {"status": "ok"}


@app.delete("/zones/{zone_id}")
def delete_zone(zone_id: int, claim=Depends(require_role("admin"))):
    db_execute("DELETE FROM zones WHERE id = ?", (zone_id,))
    db_commit()
    audit(claim["sub"], "delete_zone", str(zone_id))
    return {"status": "ok"}


@app.get("/config")
def get_config():
    # Return in-memory config for simplicity
    return {"config": CONFIG}


@app.post("/config")
def update_config(update: ConfigUpdate, claim=Depends(require_role("admin"))):
    if update.ppe_classes is not None:
        CONFIG["ppe_classes"].update({k: v for k, v in update.ppe_classes.items() if k in CONFIG["ppe_classes"]})
    if update.rules is not None:
        CONFIG["rules"].update(update.rules)
    if update.api_key:
        CONFIG["api_key"] = update.api_key
    if update.alert_webhook is not None:
        CONFIG["alert_webhook"] = update.alert_webhook
    if update.smtp is not None:
        CONFIG["smtp"].update(update.smtp)
    if update.privacy is not None:
        CONFIG["privacy"].update(update.privacy)
    if update.alerts is not None:
        CONFIG["alerts"].update(update.alerts)
    if update.jwt is not None:
        CONFIG["jwt"].update(update.jwt)
    if update.twilio is not None:
        CONFIG["twilio"].update(update.twilio)
    audit(claim["sub"], "update_config")
    return {"config": CONFIG}


@app.post("/alert/test")
def alert_test(claim=Depends(require_role("admin"))):
    trigger_alert({"type": "test", "ts": time.time(), "source": "system"})
    audit(claim["sub"], "alert_test")
    return {"status": "sent"}


# ---------- Model mgmt & Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


class ModelReload(BaseModel):
    weights: str


@app.post("/model/reload")
def model_reload(body: ModelReload, claim=Depends(require_role("admin"))):
    ok = load_model_from(body.weights)
    audit(claim["sub"], "model_reload", body.weights)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to load weights")
    return {"status": "ok", "weights": body.weights}


@app.get("/model")
def model_info():
    try:
        names = None
        if hasattr(model, "names"):
            names = getattr(model, "names")
        return {
            "loaded": model is not None,
            "thresholds": CONFIG.get("thresholds"),
            "classes": names,
        }
    except Exception:
        return {"loaded": model is not None, "thresholds": CONFIG.get("thresholds")}


# ---------- Auth & Users ----------
@app.post("/auth/login")
def login(req: LoginRequest):
    rows = db_fetchall("SELECT username, password_hash, role FROM users WHERE username = ?", (req.username,))
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    username, pw_hash, role = rows[0]
    if not pbkdf2_sha256.verify(req.password, pw_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = new_jwt(username, role)
    audit(username, "login")
    return {"access_token": token, "token_type": "bearer", "role": role}


@app.post("/auth/register")
def register(user: UserCreate, claim=Depends(require_role("admin"))):
    db_execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)", (user.username, pbkdf2_sha256.hash(user.password), user.role))
    db_commit()
    audit(claim["sub"], "register_user", user.username)
    return {"status": "ok"}


@app.get("/users")
def list_users(_: dict = Depends(require_role("admin"))):
    rows = db_fetchall("SELECT id, username, role FROM users ORDER BY id DESC")
    return {"users": [{"id": r[0], "username": r[1], "role": r[2]} for r in rows]}


# ---------- Analytics & Reports ----------
@app.get("/analytics/summary")
def analytics_summary(days: int = 7):
    since = time.time() - days * 86400
    rows = db_fetchall("SELECT event_type, COUNT(*) FROM events WHERE ts > ? GROUP BY event_type", (since,))
    by_type = {r[0]: r[1] for r in rows}
    # daily trend
    rows2 = db_fetchall("SELECT CAST(ts/86400 AS INT) as d, COUNT(*) FROM events WHERE ts > ? GROUP BY d ORDER BY d ASC", (since,))
    trend = [{"day": int(r[0]), "count": int(r[1])} for r in rows2]
    return {"by_type": by_type, "trend": trend}


@app.get("/reports/events.csv")
def report_events_csv():
    rows = db_fetchall("SELECT ts, source, event_type, class_id, conf, x1, y1, x2, y2, track_id FROM events ORDER BY id DESC LIMIT 1000")
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["ts", "source", "type", "class_id", "conf", "x1", "y1", "x2", "y2", "track_id"])
    for r in rows:
        writer.writerow(r)
    return Response(content=out.getvalue(), media_type="text/csv")


@app.get("/reports/summary.pdf")
def report_summary_pdf():
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        raise HTTPException(status_code=500, detail="PDF generator not available")
    rows = db_fetchall("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    w, h = A4
    y = h - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "Safety Report Summary")
    y -= 24
    c.setFont("Helvetica", 12)
    for r in rows:
        c.drawString(72, y, f"{r[0]}: {r[1]}")
        y -= 18
    c.showPage()
    c.save()
    pdf = buff.getvalue()
    return Response(content=pdf, media_type="application/pdf")


# ---------- Metrics & Retention ----------
EVENTS_COUNTER = Counter("events_total", "Total events by type", ["event_type"])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def retention_task():
    while True:
        try:
            days = int(CONFIG["privacy"].get("retention_days", 30) or 30)
            cutoff = time.time() - days * 86400
            db_execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            db_commit()
        except Exception:
            pass
        await asyncio.sleep(12 * 3600)

if __name__ == "__main__":
    import uvicorn

    # Run with the app object to avoid double-import (and Prometheus duplication) when executing app.py directly.
    uvicorn.run(app, host="0.0.0.0", port=8000)
