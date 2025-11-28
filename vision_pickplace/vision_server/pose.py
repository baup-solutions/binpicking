# vision_server/pose.py
import math
from typing import List, Tuple, Optional, Dict

import numpy as np


def disparity_to_depth(disparity: np.ndarray, scale: float, offset: float, baseline: float, fx: float) -> np.ndarray:
    """
    Convierte imagen de disparidad (entera) a profundidad Z (m).
    La relación exacta está en la doc del rc_visard: Z = f(baseline, disparity, intrínsecas).
    Aquí usamos un modelo típico Z = fx * baseline / (scale*disp + offset).
    Ajusta scale/offset según el archivo de parámetros de discriminación. :contentReference[oaicite:8]{index=8}
    """
    disp = disparity.astype(np.float32) * scale + offset
    valid = disp > 0
    Z = np.zeros_like(disp, dtype=np.float32)
    Z[valid] = fx * baseline / disp[valid]
    Z[~valid] = 0.0
    return Z


def bbox_to_3d_pose(
    bbox: List[float],
    depth_img: np.ndarray,
    intrinsics: Dict[str, float],
    T_base_cam: np.ndarray,
    min_points: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    bbox: [x1,y1,x2,y2] en pixeles
    depth_img: profundidad en metros (H x W)
    intrinsics: {fx,fy,cx,cy}
    T_base_cam: 4x4
    Devuelve (T_base_obj, centroid_cam), donde T_base_obj es 4x4.
    """
    h, w = depth_img.shape
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    roi = depth_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    Z = roi.copy()
    Z[Z <= 0] = np.nan
    ys, xs = np.where(~np.isnan(Z))
    if len(xs) < min_points:
        return None

    # muestreamos para eficiencia
    if len(xs) > 5000:
        idx = np.random.choice(len(xs), 5000, replace=False)
        xs, ys = xs[idx], ys[idx]

    zs = Z[ys, xs]

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    u = xs + x1
    v = ys + y1

    X = (u - cx) * zs / fx
    Y = (v - cy) * zs / fy

    pts_cam = np.vstack((X, Y, zs)).T  # Nx3

    centroid_cam = pts_cam.mean(axis=0)
    pts_centered = pts_cam - centroid_cam

    # PCA para orientación de la caja en marco de cámara
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R_cam_obj = eigvecs[:, order]  # columnas = ejes principales

    # Aseguramos que R es una rotación (det>0)
    if np.linalg.det(R_cam_obj) < 0:
        R_cam_obj[:, 0] *= -1

    # Construimos T_cam_obj
    T_cam_obj = np.eye(4, dtype=np.float32)
    T_cam_obj[:3, :3] = R_cam_obj
    T_cam_obj[:3, 3] = centroid_cam

    # Transformamos a base del robot: T_base_obj = T_base_cam * T_cam_obj
    T_base_obj = T_base_cam @ T_cam_obj

    return T_base_obj, centroid_cam


def rotmat_to_abc(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convierte matriz de rotación R (3x3) a ángulos ZYX (A,B,C) en grados.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        A = math.atan2(R[1, 0], R[0, 0])
        B = math.atan2(-R[2, 0], sy)
        C = math.atan2(R[2, 1], R[2, 2])
    else:
        A = math.atan2(-R[0, 1], R[1, 1])
        B = math.atan2(-R[2, 0], sy)
        C = 0.0

    return math.degrees(A), math.degrees(B), math.degrees(C)
