# vision_server/main.py
import argparse
import json
import threading
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
import zmq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from ultralytics import YOLO

from camera_client import ZmqFrameSubscriber
from pose import bbox_to_3d_pose, rotmat_to_abc

import socketserver


class Config:
    def __init__(self, path: Path):
        self.path = path
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)

    def save(self):
        with open(self.path, "w") as f:
            yaml.safe_dump(self.data, f)

    @property
    def camera(self) -> Dict[str, Any]:
        return self.data["camera"]

    @property
    def intrinsics(self) -> Dict[str, float]:
        return self.data["intrinsics"]

    @property
    def extrinsics(self) -> Dict[str, Any]:
        return self.data["extrinsics"]

    @property
    def models(self) -> Dict[str, Any]:
        return self.data["models"]


# ---------- Cargar config ----------

BASE_DIR = Path(__file__).resolve().parent
cfg = Config(BASE_DIR / "config.yaml")

# Matriz T_base_cam
T_base_cam = np.array(cfg.extrinsics["T_base_cam"], dtype=np.float32)

# Suscriptores ZMQ
rgb_sub = ZmqFrameSubscriber(cfg.camera["rgb_zmq"], name="rgb")
depth_sub = ZmqFrameSubscriber(cfg.camera["depth_zmq"], name="depth")
rgb_sub.start()
depth_sub.start()

# Cargar modelos YOLO
detectors: Dict[str, YOLO] = {}
for model_id, m in cfg.models.items():
    weights = BASE_DIR / m["path"]
    detectors[model_id] = YOLO(str(weights))


# ---------- API FastAPI ----------

app = FastAPI(title="Vision Pick&Place")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# Servimos la web estática desde ../webui
app.mount(
    "/web",
    StaticFiles(directory=str(BASE_DIR.parent / "webui"), html=True),
    name="web",
)


@app.get("/api/models")
def list_models():
    return [
        {
            "id": mid,
            "conf": cfg.models[mid]["conf"],
            "iou": cfg.models[mid]["iou"],
        }
        for mid in cfg.models.keys()
    ]


@app.post("/api/models/{model_id}")
def update_model(model_id: str, body: Dict[str, Any]):
    if model_id not in cfg.models:
        return {"success": False, "message": "Modelo desconocido"}
    m = cfg.models[model_id]
    if "conf" in body:
        m["conf"] = float(body["conf"])
    if "iou" in body:
        m["iou"] = float(body["iou"])
    cfg.save()
    return {"success": True, "message": "Actualizado", "model": m}


@app.get("/api/last_frame_info")
def last_frame_info():
    rgb, h_rgb = rgb_sub.latest()
    depth, h_dep = depth_sub.latest()
    return {
        "rgb": h_rgb,
        "depth": h_dep,
    }


@app.post("/api/detect/{model_id}")
def detect(model_id: str):
    if model_id not in detectors:
        return {"success": False, "message": f"Modelo {model_id} no existe"}

    rgb, rgb_info = rgb_sub.latest()
    depth, depth_info = depth_sub.latest()

    if rgb is None or depth is None:
        return {"success": False, "message": "No hay frames aún"}

    model_cfg = cfg.models[model_id]
    model = detectors[model_id]

    conf = model_cfg["conf"]
    iou = model_cfg["iou"]

    # YOLO espera imagen RGB (HWC)
    # Si la cámara envía BGR, invierte aquí; con Aravis/rc_visard es típico RGB.
    results = model(rgb, conf=conf, iou=iou)
    if not results:
        return {"success": False, "message": "No hay detecciones"}

    det = results[0]
    if det.boxes is None or det.boxes.shape[0] == 0:
        return {"success": False, "message": "No hay detecciones"}

    # Cogemos la detección con mayor score
    scores = det.boxes.conf.cpu().numpy()
    boxes = det.boxes.xyxy.cpu().numpy()
    if "class_name" in model_cfg and model_cfg["class_name"]:
        # filtrar por clase si hiciste entrenamiento con varias clases
        cls_ids = det.boxes.cls.cpu().numpy().astype(int)
        # aquí deberías mapear class_name -> id; para simplificar, cogemos todas
    best_idx = int(np.argmax(scores))
    best_bbox = boxes[best_idx].tolist()
    best_score = float(scores[best_idx])

    # depth_img: si es disparity, conviértela a depth antes.
    depth_img_m = depth.astype(np.float32) / 1000.0  # ej: si viene en mm

    pose = bbox_to_3d_pose(
        best_bbox,
        depth_img_m,
        cfg.intrinsics,
        T_base_cam,
        min_points=model_cfg.get("min_points", 200),
    )
    if pose is None:
        return {"success": False, "message": "No se pudo calcular pose 3D"}

    T_base_obj, centroid_cam = pose
    R = T_base_obj[:3, :3]
    t = T_base_obj[:3, 3]
    A, B, C = rotmat_to_abc(R)

    return {
        "success": True,
        "bbox": best_bbox,
        "score": best_score,
        "pose_base": {
            "x": float(t[0]),
            "y": float(t[1]),
            "z": float(t[2]),
            "A": A,
            "B": B,
            "C": C,
        },
        "debug": {
            "centroid_cam": [float(c) for c in centroid_cam],
        },
    }


# ---------- Servidor TCP para KUKA ----------

class KukaHandler(socketserver.StreamRequestHandler):
    """
    Protocolo textual sencillo:
    KUKA envía: "GET_POSE box\n"
    Servidor responde:
      OK x y z A B C\n
    o
      ERROR <mensaje>\n
    """

    def handle(self):
        line = self.rfile.readline().decode("utf-8").strip()
        if not line:
            return
        parts = line.split()
        if len(parts) != 2 or parts[0] != "GET_POSE":
            self.wfile.write(b"ERROR bad_command\n")
            return

        model_id = parts[1]
        # reutilizamos la lógica de /api/detect
        from fastapi.testclient import TestClient  # truco sencillo

        client = TestClient(app)
        resp = client.post(f"/api/detect/{model_id}")
        data = resp.json()

        if not data.get("success", False):
            msg = data.get("message", "fail")
            self.wfile.write(f"ERROR {msg}\n".encode("utf-8"))
            return

        pose = data["pose_base"]
        s = "OK {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f}\n".format(
            pose["x"], pose["y"], pose["z"], pose["A"], pose["B"], pose["C"]
        )
        self.wfile.write(s.encode("utf-8"))


class KukaServer(threading.Thread):
    def __init__(self, host: str = "0.0.0.0", port: int = 30010):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server: socketserver.TCPServer = socketserver.TCPServer(
            (host, port), KukaHandler
        )

    def run(self):
        print(f"[KukaServer] Escuchando en {self.host}:{self.port}")
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--kuka-port", type=int, default=30010)
    args = parser.parse_args()

    kuka_srv = KukaServer(port=args.kuka-port if hasattr(args, "kuka-port") else args.kuka_port)
    # el truco anterior es feo, mejor:
    # kuka_srv = KukaServer(port=args.kuka_port)
    kuka_srv = KukaServer(port=args.kuka_port)
    kuka_srv.start()

    uvicorn.run(app, host="0.0.0.0", port=args.api_port)


if __name__ == "__main__":
    main()
