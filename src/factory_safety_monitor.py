# -*- coding: utf-8 -*-
"""
SAFETY CAM - camera-based workplace safety monitor
==================================================

This one file is the whole program (the .exe is built from it with
PyInstaller - see factory_safety_monitor.spec / build_exe.bat). Helper
scripts next to it: train_models.py (trains the YOLO models), export_onnx.py
(converts them to the .onnx files this program runs).

HOW IT WORKS (one frame of one camera)
  1. Read the camera (webcam / IP / RTSP on its own thread, or a video file).
  2. Resize to a fixed analysis size (PROC_W x PROC_H).
  3. PersonDetector (own thread) finds people with yolov8n.
  4. process_cam_frame(): tracks each person, checks which zones they are in
     and asks the PPE worker (own thread) to run the detector models on a
     crop of that person - helmet, vest, goggles, gloves, shoes, harness,
     lab coat, smoking - whatever that zone was set to check.
  5. Violations -> on-screen label, Admin Log, Telegram photo, daily alerts
     folder; RoomWatch checks lights / people after shift hours.
  6. The annotated frame is recorded (DVR) and the tile is drawn in the grid.

SECTIONS (search for the "# ── NAME" headers)
  CONFIG ........................ paths, model file names, timing constants
  CRISP GLYPH ICONS / UI TEXT ... Windows icon font + Segoe UI text for the UI
  TELEGRAM ...................... alert messages and photos (background threads)
  CHECK ITEMS + MODELS .......... what can be checked; model class -> item map
  MODEL RUNNER (ONNX Runtime) ... OnnxYOLO: runs the models on GPU (DirectML) or CPU
  ZONE TYPES / ZONE EDITOR ...... inspection / trigger / rest-area zones, drawing,
                                  editing, turning zones off for N hours
  SAFETY CHECK .................. check_items(): the model "vote" for one person
  PPE WORKER .................... runs those checks off the video thread
  TRACKING / ADMIN LOG .......... person IDs between frames; the event log
  TOP BAR / REVIEW BAR / PANELS . everything drawn around the cameras
  SHIFT & AFTER-HOURS ROOM WATCH  lights / people alerts outside shift hours
  PROCESS FRAME ................. PersonDetector + process_cam_frame()
  SETUP GUI ..................... the first window (cameras, Telegram, remote)
  REMOTE ACCESS / WEB EVENTS .... web dashboard + Cloudflare link
  DAILY ALERTS .................. alert photos saved per day (Admin Log tab)
  RECORDING (DVR) ............... continuous recording, playback, clean-up
  MAIN MONITOR .................. run_monitor(): the main loop, mouse, keys
  ENTRY ......................... program start
"""
import cv2
import numpy as np
import threading
import onnxruntime as ort
import os, sys, json, time, re, shutil, subprocess, atexit, queue, struct, bisect, ast
from datetime import datetime
from collections import defaultdict, deque
import requests
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Remote access (local web dashboard + Cloudflare Tunnel) is optional — the
# core safety-monitoring app must keep working even if Flask isn't installed.
try:
    from flask import Flask, request, session, redirect, url_for, Response, render_template_string
    from werkzeug.security import generate_password_hash, check_password_hash
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    generate_password_hash = check_password_hash = None

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG_FILE   = "system_config.json"
LOG_PATH      = "logs"
os.makedirs(LOG_PATH, exist_ok=True)

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PERSON_MODEL_PATH   = os.path.join(BASE_DIR, "yolov8n.onnx")
ICON_PATH           = os.path.join(BASE_DIR, "icon.ico")
# Every detector found here is loaded; what each one can check (helmet,
# goggles, smoking...) is worked out from its own class names via
# CLASS_ITEM_MAP, so a newly trained model only needs its file name added.
# The models are trained with ultralytics (train_models.py -> .pt) and
# exported to ONNX (export_onnx.py) - the app itself runs them with ONNX
# Runtime, so it needs neither PyTorch nor an NVIDIA card (see OnnxYOLO).
DETECTOR_MODEL_FILES = ["best_ppe.onnx", "best_ppe_v2.onnx", "best_ppe_plus.onnx", "best_smoking.onnx",
                        "best_helmet.onnx", "best_labcoat.onnx", "best_gloves.onnx"]

TELEGRAM_TOKEN      = ""
TELEGRAM_CHAT_ID    = ""
GRACE_PERIOD        = 10
PPE_INTERVAL        = 1.0
TELEGRAM_PPE_DELAY  = 5

THEME = {
    # Neutral charcoal + material-blue accent — matches the NVR-style
    # reference the user provided, instead of the earlier warm brown/amber
    # industrial palette.
    "bg_panel":      (20, 18, 18),
    "bg_panel_alt":  (30, 26, 26),
    "bg_header":     (34, 30, 30),
    "border":        (58, 55, 55),
    "accent":        (243, 150, 33),
    "accent_dim":    (150, 90, 20),
    "text_primary":  (235, 230, 230),
    "text_secondary":(155, 150, 150),
    "text_dim":      (105, 100, 100),
    "ok":            (120, 200, 90),
    "warn":          (60, 180, 255),
    "danger":        (70, 70, 235),
    "white":         (255, 255, 255),
    "black":         (0, 0, 0),
}

def rounded_rect(img, pt1, pt2, color, thickness=-1, r=8):
    x1, y1 = pt1; x2, y2 = pt2
    r = max(0, min(r, (x2-x1)//2, (y2-y1)//2))
    if thickness < 0:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, -1)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, -1)
        cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, color, -1)
        cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, color, thickness)

def text_centered(img, text, cx, y, font_scale, color, thickness=1,
                  font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = text_size(text, font, font_scale, thickness)
    put_text(img, text, (int(cx - tw/2), y), font, font_scale, color, thickness,
                cv2.LINE_AA)

def icon_camera(img, cx, cy, s, color, thickness=1):
    bw, bh = int(s*1.6), int(s*1.1)
    x1, y1 = int(cx-bw/2), int(cy-bh/2)
    rounded_rect(img, (x1, y1), (x1+bw, y1+bh), color, thickness, r=max(2, int(s*0.18)))
    cv2.rectangle(img, (int(cx-s*0.35), y1-int(s*0.28)), (int(cx+s*0.15), y1), color, thickness)
    cv2.circle(img, (cx, cy+int(s*0.05)), max(2, int(s*0.32)), color, thickness)

def icon_play(img, cx, cy, s, color):
    pts = np.array([[cx-int(s*0.4), cy-int(s*0.55)],
                     [cx-int(s*0.4), cy+int(s*0.55)],
                     [cx+int(s*0.55), cy]], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, color)

def icon_pause(img, cx, cy, s, color):
    w = max(2, int(s*0.28)); h = int(s*1.1); gap = int(s*0.22)
    cv2.rectangle(img, (cx-gap-w, cy-h//2), (cx-gap, cy+h//2), color, -1)
    cv2.rectangle(img, (cx+gap, cy-h//2), (cx+gap+w, cy+h//2), color, -1)

def icon_helmet(img, cx, cy, s, color, thickness=2):
    cv2.ellipse(img, (cx, cy), (int(s*0.55), int(s*0.4)), 0, 180, 360, color, thickness)
    cv2.line(img, (cx-int(s*0.55), cy), (cx+int(s*0.55), cy), color, thickness)
    cv2.line(img, (cx-int(s*0.12), cy-int(s*0.4)), (cx-int(s*0.12), cy-int(s*0.62)), color, thickness)

def icon_alert(img, cx, cy, s, color, thickness=2):
    pts = np.array([[cx, cy-int(s*0.6)], [cx-int(s*0.55), cy+int(s*0.45)],
                     [cx+int(s*0.55), cy+int(s*0.45)]], dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness)
    cv2.line(img, (cx, cy-int(s*0.15)), (cx, cy+int(s*0.1)), color, thickness)
    cv2.circle(img, (cx, cy+int(s*0.3)), 1, color, thickness)

def icon_rest(img, cx, cy, s, color, thickness=2):
    cv2.ellipse(img, (cx-int(s*0.05), cy), (int(s*0.4), int(s*0.35)), 0, 20, 340, color, thickness)
    cv2.ellipse(img, (cx+int(s*0.32), cy-int(s*0.05)), (int(s*0.16), int(s*0.2)), 0, -60, 140, color, thickness)

def icon_edit(img, cx, cy, s, color, thickness=1):
    p1 = (cx-int(s*0.45), cy+int(s*0.45)); p2 = (cx+int(s*0.45), cy-int(s*0.45))
    cv2.line(img, p1, p2, color, thickness+1)
    cv2.circle(img, p2, 2, color, -1)

# ── CRISP GLYPH ICONS (top bar) ────────────────────────────────────────────────
# Windows ships a real icon font (Segoe Fluent Icons / SegoeIcons.ttf) with
# clean, professional glyphs — used instead of hand-drawn cv2 shapes for the
# top bar, so those specific icons look like real pictures, not primitives.
# No download, no license to track: it's a stock OS font. Codepoints below
# were confirmed by rendering a full glyph sheet and visually matching them.
try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    _ICON_FONT_PATH = r"C:\Windows\Fonts\SegoeIcons.ttf"
    GLYPH_ICONS_AVAILABLE = os.path.exists(_ICON_FONT_PATH)
except ImportError:
    GLYPH_ICONS_AVAILABLE = False

GLYPH_CODES = {
    "log":    0xE7C3,  # page outline — Admin Log
    "select": 0xE9D5,  # checklist — Select Inspection Type
    "tools":  0xE90F,  # wrench — Tools Summary
    "edit":   0xE70F,  # pencil — Edit Mode
    "help":   0xE9CE,  # circled question mark — Help
    "pause":  0xE769,  # Pause
    "play":   0xE768,  # Play / Resume
    "camera": 0xE722,  # Camera — snapshot
    "stop":    0xE8A8,  # solid filled square — Stop (review bar; E76A looked right at
                         # large size but turned out to be a hollow outline, illegible
                         # at icon size - re-verified by rendering both at actual size)
    "rewind":  0xE892,  # skip-back — Rewind (review bar)
    "forward": 0xE893,  # skip-ahead — Forward (review bar)
    "shift":   0xE823,  # clock — Shift & Room Watch
    "bulb":    0xE82F,  # lightbulb — room lights (tile badge)
    "people":  0xE716,  # people — room occupancy (tile badge)
    "back":    0xE72B,  # left arrow — back to the setup screen
    "grid":    0xE80A,  # grid — leave the enlarged-camera view
    "video":   0xE714,  # video camera — tile header
    "shield":  0xEA18,  # shield — inspection zone / app badge
    "power":   0xE7E8,  # power — pause / resume ALL zones
    "flag":    0xE7C1,  # flag — trigger zone
    "cup":     0xEC32,  # cup — rest area
    "warning": 0xE7BA,  # warning triangle
    "expand":  0xE740,  # expand — enlarge the selected camera
    "back10":  0xED3C,  # circular arrow with "10" — back 10 s
    "fwd10":   0xED3C,  # same glyph with the arrow mirrored (see get_glyph_icon) — forward 10 s
    "frew":    0xEB9E,  # fast rewind
    "ffwd":    0xEB9D,  # fast forward
}
_glyph_cache = {}

def get_glyph_icon(name, size, color_bgr):
    """Renders (and caches) one glyph as a BGRA numpy array. Rendered once
    per (name, size, color) the first time it's needed, not every frame —
    PIL text rendering isn't free, and the top bar redraws every loop tick."""
    key = (name, size, color_bgr)
    cached = _glyph_cache.get(key)
    if cached is not None: return cached
    if not GLYPH_ICONS_AVAILABLE: return None
    try:
        font = _PILFont.truetype(_ICON_FONT_PATH, int(size*0.82))
        img = _PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
        d = _PILDraw.Draw(img)
        ch = chr(GLYPH_CODES[name])
        bbox = d.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        d.text((size/2 - tw/2 - bbox[0], size/2 - th/2 - bbox[1]), ch,
               font=font, fill=(r, g, b, 255))
        arr = np.array(img)  # RGBA
        bgra = arr[:, :, [2, 1, 0, 3]].copy()
        if name == "fwd10":
            # The icon font only has "forward 30"; mirror the arc of
            # "back 10" (top ~42%) and keep its "10" digits as they are.
            split = int(size * 0.42)
            bgra[:split] = bgra[:split, ::-1].copy()
        _glyph_cache[key] = bgra
        return bgra
    except Exception as e:
        print(f"[ICON RENDER ERROR] {e}")
        return None

def draw_glyph_icon(frame, name, cx, cy, size, color_bgr):
    """Alpha-composites a cached glyph icon centered at (cx, cy). Falls back
    to a plain circle if the icon font isn't available on this machine, so a
    missing font file never breaks the UI, just makes it a bit plainer."""
    icon = get_glyph_icon(name, size, color_bgr)
    if icon is None:
        cv2.circle(frame, (cx, cy), size//3, color_bgr, 2)
        return
    h, w = icon.shape[:2]
    _blit_bgra(frame, icon, cx - w//2, cy - h//2)

def _blit_bgra(frame, patch, x1, y1):
    h, w = patch.shape[:2]
    fh, fw = frame.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(fw, x1 + w), min(fh, y1 + h)
    if sx1 >= sx2 or sy1 >= sy2: return
    src = patch[sy1-y1:sy2-y1, sx1-x1:sx2-x1]
    roi = frame[sy1:sy2, sx1:sx2]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (roi.astype(np.float32) * (1-alpha) + src[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

# ── CRISP UI TEXT ─────────────────────────────────────────────────────────────
# OpenCV's built-in Hershey font is thin and jagged; all UI text is drawn
# with Segoe UI instead. put_text/text_size take the same arguments as
# cv2.putText/cv2.getTextSize (font scale is mapped to a pixel size with
# about the same cap height), so layout code didn't have to change. Each
# (text, size, colour) is rendered once and cached - the UI redraws every
# frame. Falls back to OpenCV's font if PIL/Segoe UI aren't there.
_UI_FONT_FILES = {False: r"C:\Windows\Fonts\segoeui.ttf", True: r"C:\Windows\Fonts\seguisb.ttf"}
UI_FONT_AVAILABLE = GLYPH_ICONS_AVAILABLE and all(os.path.exists(f) for f in _UI_FONT_FILES.values())
UI_PX_PER_SCALE = 32
_ui_fonts = {}
_text_cache = {}

def _ui_font(scale, thickness):
    key = (max(8, int(round(scale * UI_PX_PER_SCALE))), thickness >= 2)
    f = _ui_fonts.get(key)
    if f is None:
        f = _PILFont.truetype(_UI_FONT_FILES[key[1]], key[0]); _ui_fonts[key] = f
    return f

def text_size(text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.4, thickness=1):
    if not UI_FONT_AVAILABLE:
        return cv2.getTextSize(text, font, scale, thickness)
    f = _ui_font(scale, thickness)
    return (int(round(f.getlength(text))), -f.getbbox("H", anchor="ls")[1]), f.getbbox("g", anchor="ls")[3]

def put_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.4, color=(255, 255, 255),
             thickness=1, line_type=cv2.LINE_AA):
    if not UI_FONT_AVAILABLE:
        cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA); return
    if not text: return
    color = tuple(int(c) for c in color[:3])
    key = (text, scale, thickness >= 2, color)
    hit = _text_cache.get(key)
    if hit is None:
        f = _ui_font(scale, thickness)
        l, t, r, b = f.getbbox(text, anchor="ls")
        if r <= l or b <= t: return
        im = _PILImage.new("RGBA", (r - l + 2, b - t + 2), (0, 0, 0, 0))
        _PILDraw.Draw(im).text((1 - l, 1 - t), text, font=f, anchor="ls",
                               fill=(color[2], color[1], color[0], 255))
        hit = (np.array(im)[:, :, [2, 1, 0, 3]].copy(), l - 1, t - 1)
        if len(_text_cache) > 4000: _text_cache.clear()
        _text_cache[key] = hit
    patch, ox, oy = hit
    _blit_bgra(img, patch, int(org[0]) + ox, int(org[1]) + oy)

# The hand-drawn zone/camera icons are replaced by the same Segoe icon font
# the top bar uses (same signature, so callers don't change); the old
# line-drawn versions stay as the fallback when the font is missing.
def _glyph_or(name, fallback, scale=2.3):
    def draw(img, cx, cy, s, color, thickness=1):
        if GLYPH_ICONS_AVAILABLE: draw_glyph_icon(img, name, int(cx), int(cy), max(12, int(s * scale)), color)
        else: fallback(img, cx, cy, s, color, thickness)
    return draw

icon_camera = _glyph_or("video", icon_camera)
icon_helmet = _glyph_or("shield", icon_helmet)
icon_alert  = _glyph_or("warning", icon_alert)
icon_rest   = _glyph_or("cup", icon_rest)
icon_edit   = _glyph_or("edit", icon_edit, scale=2.6)

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
last_telegram_time = {}
_telegram_lock = threading.Lock()

def _send_text_worker(token, chat_id, msg):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg},
            timeout=10)
        # requests.post() only raises on connection-level failures (DNS,
        # timeout) - an HTTP 200 with {"ok": false} (wrong chat_id, bot not
        # in the chat, etc.) was silently treated as success before this
        # check existed, printing "SENT" even when Telegram rejected it.
        data = r.json()
        if data.get("ok"):
            print(f"[TELEGRAM SENT] {msg}")
        else:
            err = data.get("description", r.text)
            print(f"[TELEGRAM ERROR] {err}")
            add_log(f"Telegram send failed: {err}", THEME["danger"])
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")
        add_log(f"Telegram send failed: {e}", THEME["danger"])

def _send_photo_worker(token, chat_id, msg, img_bytes):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": msg},
            files={"photo": ("violation.jpg", img_bytes, "image/jpeg")},
            timeout=15)
        data = r.json()
        if data.get("ok"):
            print(f"[TELEGRAM PHOTO SENT] {msg}")
        else:
            err = data.get("description", r.text)
            print(f"[TELEGRAM PHOTO ERROR] {err}")
            add_log(f"Telegram photo failed: {err}", THEME["danger"])
    except Exception as e:
        print(f"[TELEGRAM PHOTO ERROR] {e}")
        add_log(f"Telegram photo failed: {e}", THEME["danger"])

def send_telegram(msg, cooldown=60):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM] {msg}"); return
    with _telegram_lock:
        key = msg[:30]; now = time.time()
        if now - last_telegram_time.get(key, 0) < cooldown: return
        last_telegram_time[key] = now
    # Fire and forget — does NOT block main thread
    t = threading.Thread(
        target=_send_text_worker,
        args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg),
        daemon=True)
    t.start()

def send_telegram_photo(msg, image, cooldown=60):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM PHOTO] {msg}"); return
    if image is None or image.size == 0:
        print(f"[TELEGRAM PHOTO] empty crop, skipped: {msg}"); return
    with _telegram_lock:
        key = msg[:30]; now = time.time()
        if now - last_telegram_time.get(key, 0) < cooldown: return
        last_telegram_time[key] = now
    try:
        ok, img_encoded = cv2.imencode(".jpg", image)
    except Exception as e:
        print(f"[TELEGRAM PHOTO ENCODE ERROR] {e}"); return
    if not ok: return
    img_bytes = img_encoded.tobytes()
    # Fire and forget — does NOT block main thread
    t = threading.Thread(
        target=_send_photo_worker,
        args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg, img_bytes),
        daemon=True)
    t.start()

def validate_telegram_config(token, chat_id, timeout=8):
    """Checks the bot token AND chat/group id against Telegram's real API
    before monitoring starts, so a typo (or a chat id that went stale after
    the group turned into a supergroup) shows up immediately as a clear
    message instead of silently failing on the first real violation, hours
    later. Returns (ok, message) - message is empty on success. Not
    configured at all (both blank) is treated as "nothing to check", not
    an error - Telegram alerts are optional."""
    if not token or not chat_id:
        return True, ""
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=timeout)
        data = r.json()
        if not data.get("ok"):
            return False, f"Telegram bot token looks invalid: {data.get('description', 'rejected by Telegram')}"
    except Exception as e:
        return False, f"Could not reach Telegram to check the bot token: {e}"
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getChat",
                          params={"chat_id": chat_id}, timeout=timeout)
        data = r.json()
        if not data.get("ok"):
            return False, (f"Telegram chat/group ID looks wrong, or the bot was never "
                            f"added to it: {data.get('description', 'rejected by Telegram')}")
    except Exception as e:
        return False, f"Could not reach Telegram to check the chat ID: {e}"
    return True, ""

def _telegram_check_worker(token, chat_id):
    """Runs on a background thread once the monitor window is already up
    and cameras are opening - the check used to block Setup's Launch button
    for several seconds before the monitor window even appeared, which felt
    like the app hanging. Now the window opens immediately and this just
    reports into the Admin Log whenever it finishes, same pattern as the
    remote-tunnel connect message."""
    ok, err = validate_telegram_config(token, chat_id)
    if ok:
        add_log("Telegram bot connected and verified", THEME["ok"])
    else:
        add_log(f"Telegram not connected: {err}", THEME["danger"])

# ── CHECK ITEMS + MODELS ──────────────────────────────────────────────────────
# What a zone can be told to check for. "ppe" items are violations when NOT
# seen on the person; "behavior" items are violations when they ARE seen.
CHECK_ITEMS = {
    "helmet":  {"label": "Helmet",  "kind": "ppe"},
    "vest":    {"label": "Vest",    "kind": "ppe"},
    "goggles": {"label": "Goggles", "kind": "ppe"},
    "gloves":  {"label": "Gloves",  "kind": "ppe"},
    "shoes":   {"label": "Shoes",   "kind": "ppe"},
    "harness": {"label": "Harness", "kind": "ppe"},
    "labcoat": {"label": "Lab Coat", "kind": "ppe"},
    "smoking": {"label": "Smoking", "kind": "behavior"},
}
DEFAULT_CHECKS = ["helmet", "vest"]
# Items that sit low on the body (or cover most of it) - the person crop must
# include the whole body, not just head/torso, when any of these are checked.
FULL_BODY_ITEMS = {"gloves", "shoes", "harness", "labcoat"}
PPE_CONF     = 0.35
# Smoking: measured on the user's lab videos (nobody smoking) the model's
# false hits topped out at 0.69 and mostly sat at 0.42-0.62 ("smoker" firing
# on faces); 0.60 removes most of them. A cigarette is small and drops in
# and out of view, so instead of demanding it on every 1s check (one miss
# would reset the alert timer), a person counts as smoking while it was seen
# in at least BEHAVIOR_MIN_HITS of their last BEHAVIOR_WINDOW checks.
SMOKING_CONF      = 0.60
BEHAVIOR_WINDOW   = 5
BEHAVIOR_MIN_HITS = 2
BEHAVIOR_RECHECK  = 0.3     # s between checks while one smoking sighting awaits confirmation

# Different datasets name the same thing differently ("Hardhat" vs "Helmet",
# "NO-Safety-Vest" vs "No_Vest"). Names are normalised (lowercase, '-'/' ' ->
# '_') and mapped to (item, positive). positive=False means "seen WITHOUT it".
CLASS_ITEM_MAP = {
    "hardhat": ("helmet", True), "helmet": ("helmet", True), "head_with_helmet": ("helmet", True),
    "no_hardhat": ("helmet", False), "no_helmet": ("helmet", False),
    "safety_vest": ("vest", True), "vest": ("vest", True),
    "no_safety_vest": ("vest", False), "no_vest": ("vest", False),
    "goggles": ("goggles", True), "safety_goggles": ("goggles", True), "eyes_with_goggles": ("goggles", True),
    "no_goggles": ("goggles", False), "eyes_without_goggles": ("goggles", False),
    "glove": ("gloves", True), "gloves": ("gloves", True),
    "no_glove": ("gloves", False), "no_gloves": ("gloves", False),
    "shoe": ("shoes", True), "shoes": ("shoes", True), "boots": ("shoes", True),
    "safety_shoe": ("shoes", True), "safety_shoes": ("shoes", True), "safety_boots": ("shoes", True),
    "no_shoe": ("shoes", False), "no_shoes": ("shoes", False), "no_boots": ("shoes", False),
    "safety_harness": ("harness", True), "harness": ("harness", True),
    "no_harness": ("harness", False),
    "lab_coat": ("labcoat", True), "labcoat": ("labcoat", True), "lab_coat_compliance": ("labcoat", True),
    "person_with_labcoat": ("labcoat", True),
    "person_without_labcoat": ("labcoat", False), "no_lab_coat": ("labcoat", False), "no_labcoat": ("labcoat", False),
    "cigarette": ("smoking", True), "smoking": ("smoking", True), "smoke": ("smoking", True),
    "smoker": ("smoking", True), "vape": ("smoking", True),
}

# Class names whose meaning depends on the dataset: in the Hard Hat Workers
# helmet dataset "head" is a head WITHOUT a helmet, while other datasets use
# "head" for any head - so it can't go in the global map.
MODEL_CLASS_OVERRIDES = {
    "best_helmet.onnx": {"head": ("helmet", False)},
}

# Models trusted for only some of the items they detect. best_gloves is our
# own model, trained on ~10k images merged from 6 datasets (merge_datasets.py:
# white work gloves, blue/teal nitrile, lab and welding gloves, bare hands).
# On hand-labelled people from our workshop videos it caught 6 of 7 gloved
# hands (the earlier models: 2 of 7) with no false "gloves" on 107 bare hands.
MODEL_ITEM_ALLOW = {
    "best_gloves.onnx": {"gloves"},
}

def _norm_class(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

detector_models = []   # [{"file", "model", "class_map": {cls_idx: (item, positive)}, "items": set}]

def supported_items():
    s = set()
    for m in detector_models: s |= m["items"]
    return s

# ── MODEL RUNNER (ONNX Runtime) ───────────────────────────────────────────────
# Runs a YOLOv8 model exported to ONNX. Gives exactly the same boxes and
# confidences as ultralytics (checked on all 8 models: 100% of boxes match,
# same scores) because it prepares the image the same way - resize to the
# model's training size, pad to a multiple of 32 with grey (114), RGB 0..1 -
# and decodes the output the same way (best class per anchor, then
# per-class non-max suppression at IoU 0.7).
# DirectML lets ONNX Runtime use ANY DirectX 12 graphics card - NVIDIA, AMD
# or a laptop's built-in Intel GPU - and it falls back to the CPU if there
# is none. That's what lets the program run on a weaker laptop, and dropping
# PyTorch shrank the installed program from 4.4 GB to well under 1 GB.
_dml_lock = threading.Lock()     # DirectML: one inference at a time on the GPU

class _Det:
    """One detection, shaped like an ultralytics box: .xyxy[0], .conf[0], .cls[0]."""
    __slots__ = ("xyxy", "conf", "cls")
    def __init__(self, xyxy, conf, cls):
        self.xyxy = (xyxy,); self.conf = (conf,); self.cls = (cls,)

class _Result:
    __slots__ = ("boxes",)
    def __init__(self, boxes): self.boxes = boxes

class OnnxYOLO:
    def __init__(self, path):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Several models run side by side (person thread + PPE thread): by
        # default each keeps all its CPU threads busy-waiting between runs,
        # starving the others - measured 50 ms -> 550 ms per model. Fewer
        # threads that sleep when idle keep every model at full speed.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.set_denormal_as_zero", "1")
        so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
        providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        if os.environ.get("SAFETYCAM_CPU") == "1":      # force CPU (testing / a GPU driver problem)
            providers = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in providers:
            so.enable_mem_pattern = False                         # required by DirectML
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self.sess = ort.InferenceSession(path, so, providers=providers)
        except Exception:                                         # no usable GPU: CPU
            self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.on_gpu = self.sess.get_providers()[0] == "DmlExecutionProvider"
        self.inp = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map    # written by the ultralytics export
        self.names = {int(k): v for k, v in ast.literal_eval(meta.get("names", "{}")).items()}
        self.size = int(ast.literal_eval(meta.get("imgsz", "[640, 640]"))[0])   # its training size

    def __call__(self, img, conf=0.25, iou=0.7, verbose=False, max_det=300):
        h, w = img.shape[:2]; S = self.size
        r = min(S / h, S / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
        dw, dh = ((S - nw) % 32) / 2, ((S - nh) % 32) / 2      # pad to a multiple of 32
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        im = cv2.copyMakeBorder(im, top, int(round(dh + 0.1)), left, int(round(dw + 0.1)),
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0, swapRB=True)          # 1x3xHxW, RGB, 0..1
        if self.on_gpu:
            with _dml_lock: out = self.sess.run(None, {self.inp: blob})[0][0]
        else:
            out = self.sess.run(None, {self.inp: blob})[0][0]
        out = out.T                                    # one row per anchor: cx, cy, w, h, class scores...
        scores = out[:, 4:]
        cls = scores.argmax(1); cf = scores[np.arange(len(cls)), cls]
        keep = cf > conf
        if not keep.any(): return [_Result([])]
        b, cls, cf = out[keep, :4], cls[keep], cf[keep]
        xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1)
        idx = np.array(cv2.dnn.NMSBoxesBatched(xywh.tolist(), cf.tolist(), cls.tolist(), conf, iou)).flatten()
        dets = []
        for i in idx[:max_det]:
            x1, y1, bw, bh = xywh[i]
            x1 = (x1 - left) / r; y1 = (y1 - top) / r          # back to the original image's pixels
            dets.append(_Det(np.array([max(0, x1), max(0, y1), min(w, x1 + bw / r), min(h, y1 + bh / r)],
                                      np.float32), float(cf[i]), float(cls[i])))
        return [_Result(dets)]

def load_models():
    if not os.path.exists(PERSON_MODEL_PATH):
        add_log(f"Person model not found at {PERSON_MODEL_PATH}", THEME["warn"])
    person_mdl = OnnxYOLO(PERSON_MODEL_PATH)
    print(f"Person model loaded on {'GPU (DirectML)' if person_mdl.on_gpu else 'CPU'}")

    detector_models.clear()
    for fname in DETECTOR_MODEL_FILES:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path): continue
        try:
            mdl = OnnxYOLO(path)
        except Exception as e:
            add_log(f"Could not load {fname}: {e}", THEME["warn"]); continue
        class_map = {}
        overrides = MODEL_CLASS_OVERRIDES.get(fname, {})
        for idx, cname in mdl.names.items():
            n = _norm_class(cname)
            hit = overrides.get(n) or CLASS_ITEM_MAP.get(n)
            if hit and hit[0] in MODEL_ITEM_ALLOW.get(fname, CHECK_ITEMS): class_map[int(idx)] = hit
        items = {item for item, _ in class_map.values()}
        if not items:
            print(f"[MODELS] {fname} has no classes this app understands: {mdl.names}")
            continue
        detector_models.append({"file": fname, "model": mdl, "class_map": class_map, "items": items})
        print(f"[MODELS] {fname}: {sorted(items)}")
    if not detector_models:
        add_log("No PPE/safety detector models found", THEME["warn"])

    # A freshly loaded model's FIRST real inference call pays a one-time cost
    # (CUDA kernel JIT compilation on GPU, thread-pool/oneDNN setup on CPU)
    # that has nothing to do with frame size or camera speed - measured at
    # ~2.5s for the first call and ~0.9s for the next model's first call.
    # Left alone, that cost silently lands on whichever real video frames
    # happen to trigger each model first, which is exactly the choppy/
    # stuttering start the user was seeing. Paying it here instead, on a
    # throwaway blank frame while the user is still on the Setup screen,
    # means every real frame in run_monitor() runs at steady-state speed
    # from frame 1.
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for m in [person_mdl] + [d["model"] for d in detector_models]:
        try: m(dummy, verbose=False)
        except Exception: pass

    return person_mdl

def set_opencv_window_icon(window_title, icon_path):
    """Sets the taskbar/titlebar icon for an OpenCV HighGUI window.
    OpenCV has no built-in API for this - this is a Windows-only
    workaround via ctypes + Win32 API. Must be called AFTER the window
    has actually been shown at least once (e.g. after the first
    cv2.imshow), otherwise FindWindowW may not locate it yet. Silently
    does nothing on non-Windows platforms or if anything goes wrong,
    since a missing icon should never crash the program."""
    if sys.platform != "win32":
        return False
    if not os.path.exists(icon_path):
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        IMAGE_ICON      = 1
        LR_LOADFROMFILE = 0x10
        LR_DEFAULTSIZE  = 0x40
        WM_SETICON      = 0x0080
        ICON_SMALL, ICON_BIG = 0, 1
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        h_icon = user32.LoadImageW(
            None, icon_path, IMAGE_ICON, 0, 0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE)
        if not h_icon:
            return False
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_icon)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG,   h_icon)
        return True
    except Exception:
        return False

# ── ZONE TYPES ────────────────────────────────────────────────────────────────
# "ppe_check" = inspect while the person is inside the zone. "ppe_trigger" =
# a doorway-style zone: passing through it arms its checks for that person
# until they pass them. Both carry their own "checks" list (any CHECK_ITEMS).
ZONE_TYPES = {
    "ppe_check":    {"label": "Inspection", "color": (80, 220, 220), "icon": icon_helmet, "has_checks": True},
    "ppe_trigger":  {"label": "Trigger",    "color": (60, 150, 255), "icon": icon_alert,  "has_checks": True},
    "rest_area":    {"label": "Rest Area",  "color": (110, 210, 110),"icon": icon_rest,   "has_checks": False},
}

# A zone can be switched off without deleting it: zone["off"] is None (on),
# "manual" (off until someone turns it back on) or a timestamp (off until
# then - e.g. 2 h for a breakdown or an event in that area).
ZONE_OFF_HOURS = [1, 2, 3, 4, 5]

# One switch for ALL zones (top bar power button), stored as config
# "zones_paused" - None, "manual" or a timestamp, like one zone's "off".
# It sits on top of each zone's own setting instead of overwriting it, so
# turning everything back on restores exactly what was on before.
ZONES_PAUSE = {"off": None}

def zones_paused(now=None):
    off = ZONES_PAUSE["off"]
    if off is None: return False
    if off == "manual": return True
    return (now or time.time()) < off

def zones_pause_text():
    off = ZONES_PAUSE["off"]
    if off is None: return ""
    if off == "manual": return "until turned on"
    return "until " + datetime.fromtimestamp(off).strftime("%I:%M %p").lstrip("0")

def zone_on(zone, now=None):
    if zones_paused(now): return False
    off = zone.get("off")
    if off is None: return True
    if off == "manual": return False
    return (now or time.time()) >= off

def zone_off_text(zone):
    if zones_paused(): return "OFF - all zones " + zones_pause_text()
    off = zone.get("off")
    if off is None: return ""
    if off == "manual": return "OFF until turned on"
    return "OFF until " + datetime.fromtimestamp(off).strftime("%I:%M %p").lstrip("0")

def draw_dashed_poly(img, pts, color, thickness=1, dash=10, gap=7):
    n = len(pts)
    for i in range(n):
        a = np.array(pts[i], np.float32); b = np.array(pts[(i+1) % n], np.float32)
        L = float(np.linalg.norm(b - a))
        if L < 1: continue
        d = (b - a) / L; t = 0.0
        while t < L:
            p1 = a + d*t; p2 = a + d*min(L, t+dash)
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)
            t += dash + gap

def ordered_checks(checks):
    return [k for k in CHECK_ITEMS if k in set(checks)]

def zone_auto_name(ztype, checks):
    if not ZONE_TYPES[ztype]["has_checks"]:
        return ZONE_TYPES[ztype]["label"]
    labels = " + ".join(CHECK_ITEMS[c]["label"] for c in ordered_checks(checks)) or "No checks"
    return labels if ztype == "ppe_check" else f"Trigger: {labels}"

def default_config():
    return {"cameras": [], "zones": {},
            "recording_enabled": True, "recording_keep_days": REC_KEEP_DAYS, "recording_max_gb": REC_MAX_GB,
            "telegram_token": "", "telegram_chat_id": "",
            "remote_enabled": False, "remote_user": "", "remote_pass_hash": "",
            "after_hours": default_after_hours()}

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f: json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[CONFIG SAVE ERROR] {e}")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: return json.load(f)
        except: pass
    return default_config()

# ── ZONE EDITOR ───────────────────────────────────────────────────────────────
# Zones are always stored in this fixed virtual coordinate space, regardless
# of what resolution the window/camera tiles are actually rendered at right
# now. Storage never changes when the window is resized — only the on-screen
# scale factor used to draw/hit-test them does. This is what lets the window
# be freely resizable while zones stay correctly aligned with the video.
ZONE_BASE_W = 640
ZONE_BASE_H = 480

class ZoneEditor:
    def __init__(self):
        self.zones          = {}
        self.selected_zone  = None
        self.selected_pt    = None
        self.selected_index = None
        self.DRAG_R         = 10

    def load(self, cfg):
        """Also migrates older configs: oscilloscope/toolbox zones are
        dropped (that inspection no longer exists), and PPE zones saved
        before per-zone checks existed keep their old behaviour (helmet +
        vest) instead of silently checking nothing."""
        self.zones = {}
        for k, v in cfg.get("zones", {}).items():
            kept = []
            for z in v:
                if z.get("type") not in ZONE_TYPES: continue
                if ZONE_TYPES[z["type"]]["has_checks"] and "checks" not in z:
                    z["checks"] = list(DEFAULT_CHECKS)
                    z["name"] = zone_auto_name(z["type"], z["checks"])
                kept.append(z)
            self.zones[int(k)] = kept

    def save_to(self, cfg):
        cfg["zones"] = {str(k): v for k, v in self.zones.items()}

    def get_zones(self, ci):
        return self.zones.get(ci, [])

    def get_selected(self):
        if self.selected_index is None or self.selected_zone is None: return None
        zones = self.get_zones(self.selected_index)
        return zones[self.selected_zone] if self.selected_zone < len(zones) else None

    def add_zone(self, ci, ztype, pts, checks=None):
        z = {"type": ztype, "points": pts, "active": True}
        if ZONE_TYPES[ztype]["has_checks"]:
            z["checks"] = ordered_checks(checks or [])
        z["name"] = zone_auto_name(ztype, z.get("checks", []))
        self.zones.setdefault(ci, []).append(z)

    def set_zone_settings(self, zone, ztype=None, checks=None):
        if ztype is not None: zone["type"] = ztype
        if ZONE_TYPES[zone["type"]]["has_checks"]:
            if checks is not None: zone["checks"] = ordered_checks(checks)
            zone.setdefault("checks", list(DEFAULT_CHECKS))
        else:
            zone.pop("checks", None)
        zone["name"] = zone_auto_name(zone["type"], zone.get("checks", []))

    def set_zone_off(self, zone, value):
        if value is None: zone.pop("off", None)
        else: zone["off"] = value

    def expire_offs(self, now):
        """Zones whose 'off for N hours' ran out - switched back on."""
        back = []
        for ci, zs in self.zones.items():
            for z in zs:
                off = z.get("off")
                if off not in (None, "manual") and now >= off:
                    z.pop("off", None); back.append((ci, z))
        return back

    def delete_zone(self, ci, zi):
        if ci in self.zones and zi < len(self.zones[ci]):
            self.zones[ci].pop(zi)

    def scaled_points(self, zone, pw=ZONE_BASE_W, ph=ZONE_BASE_H):
        """Zone points converted from the fixed storage space to whatever
        pixel size (pw, ph) is actually being rendered/detected on right now."""
        sx, sy = pw/ZONE_BASE_W, ph/ZONE_BASE_H
        return np.array([[int(p[0]*sx), int(p[1]*sy)] for p in zone["points"]], dtype=np.int32)

    def pt_in_zone(self, zone, x, y, pw=ZONE_BASE_W, ph=ZONE_BASE_H):
        pts = self.scaled_points(zone, pw, ph)
        if len(pts) < 3: return False
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    def box_in_zone(self, zone, bx1, by1, bx2, by2, samples=5, pw=ZONE_BASE_W, ph=ZONE_BASE_H):
        pts = self.scaled_points(zone, pw, ph)
        if len(pts) < 3: return False
        xs = np.linspace(bx1, bx2, samples)
        ys = np.linspace(by1, by2, samples)
        for sx in xs:
            for sy in ys:
                if cv2.pointPolygonTest(pts, (float(sx), float(sy)), False) >= 0:
                    return True
        return False

    def select_zone_by_index(self, ci, zi):
        zones = self.get_zones(ci)
        if 0 <= zi < len(zones):
            self.selected_zone  = zi
            self.selected_index = ci
            self.selected_pt    = None
            return True
        return False

    def draw_zones(self, frame, ci, edit_mode=False, pw=ZONE_BASE_W, ph=ZONE_BASE_H):
        zones = self.get_zones(ci)
        # One frame-copy + one blend for ALL zone fills instead of one pair
        # per zone (used to be O(zone_count) full-frame copies every frame,
        # per camera) — same visual result, cheaper with several zones.
        fillable = [z for z in zones if len(z["points"]) >= 3 and zone_on(z)]
        if fillable:
            ov = frame.copy()
            for zone in fillable:
                pts = self.scaled_points(zone, pw, ph)
                col = ZONE_TYPES.get(zone["type"], {}).get("color", THEME["text_secondary"])
                cv2.fillPoly(ov, [pts], col)
            cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
        for zi, zone in enumerate(zones):
            pts = self.scaled_points(zone, pw, ph)
            col = ZONE_TYPES.get(zone["type"], {}).get("color", THEME["text_secondary"])
            on = zone_on(zone)
            if not on: col = THEME["text_dim"]
            if len(pts) >= 3:
                if on: cv2.polylines(frame, [pts], True, col, 2)
                else:  draw_dashed_poly(frame, pts.tolist(), col, 2)
            if len(pts) > 0:
                put_text(frame, zone["name"] if on else f"{zone['name']}  (OFF)",
                            (pts[0][0], pts[0][1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
            if (edit_mode and self.selected_index == ci
                    and self.selected_zone == zi and len(pts) >= 3):
                cv2.polylines(frame, [pts], True, THEME["white"], 3)
            if edit_mode:
                for p in pts:
                    cv2.circle(frame, tuple(p), 7, THEME["danger"], -1)
                    cv2.circle(frame, tuple(p), 7, THEME["white"], 1)

    def seg_dist(self, p, a, b):
        p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
        ab = b - a; t = np.dot(p-a, ab) / (np.dot(ab, ab) + 1e-6)
        return np.linalg.norm(p - (a + max(0, min(1, t)) * ab))

    def select_zone_at(self, ci, x, y):
        for zi, zone in enumerate(self.get_zones(ci)):
            if self.pt_in_zone(zone, x, y):
                self.selected_zone  = zi
                self.selected_pt    = None
                self.selected_index = ci
                return True
        self.selected_zone  = None
        self.selected_index = None
        return False

    def resize_selected(self, ci, scale):
        if self.selected_index != ci or self.selected_zone is None: return
        zones = self.get_zones(ci)
        if self.selected_zone >= len(zones): return
        pts    = np.array(zones[self.selected_zone]["points"], dtype=float)
        center = pts.mean(axis=0)
        pts    = center + (pts - center) * scale
        zones[self.selected_zone]["points"] = pts.astype(int).tolist()

    def delete_selected(self, ci):
        if self.selected_index == ci and self.selected_zone is not None:
            self.delete_zone(ci, self.selected_zone)
            self.selected_zone  = None
            self.selected_index = None

    def deselect(self):
        self.selected_zone  = None
        self.selected_pt    = None
        self.selected_index = None

    def handle_mouse(self, event, x, y, ci, pw=ZONE_BASE_W, ph=ZONE_BASE_H):
        # (x, y) arrive in real on-screen pixels for whatever size the tile
        # is rendered at right now — convert once here so everything below
        # keeps comparing/storing in the fixed virtual space, unchanged.
        x, y = int(x*ZONE_BASE_W/pw), int(y*ZONE_BASE_H/ph)
        zones = self.get_zones(ci)
        DR    = self.DRAG_R
        if event == cv2.EVENT_LBUTTONDOWN:
            for zi, zone in enumerate(zones):
                for pi, p in enumerate(zone["points"]):
                    if abs(p[0]-x) < DR and abs(p[1]-y) < DR:
                        self.selected_zone = zi; self.selected_pt = pi; return
            for zi, zone in enumerate(zones):
                n = len(zone["points"])
                for i in range(n):
                    if self.seg_dist((x, y), zone["points"][i],
                                     zone["points"][(i+1) % n]) < 8:
                        zone["points"].insert(i+1, [x, y])
                        self.selected_zone = zi; self.selected_pt = i+1; return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selected_zone is not None and self.selected_pt is not None:
                if (0 <= self.selected_zone < len(zones)
                        and 0 <= self.selected_pt < len(zones[self.selected_zone]["points"])):
                    zones[self.selected_zone]["points"][self.selected_pt] = [x, y]
                else:
                    self.selected_zone = None; self.selected_pt = None
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_zone = None; self.selected_pt = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.select_zone_at(ci, x, y): return
            for zi, zone in enumerate(zones):
                for pi, p in enumerate(zone["points"]):
                    if abs(p[0]-x) < DR and abs(p[1]-y) < DR and len(zone["points"]) > 3:
                        zone["points"].pop(pi); return

# ── SAFETY CHECK ──────────────────────────────────────────────────────────────
def check_items(frame, x1, y1, x2, y2, items):
    """Runs only the detector models that can check at least one of the
    requested items, on a crop of this person, and returns {item: ok} for
    every requested item some loaded model supports. Items no model supports
    are left out entirely - they must never count as a violation just
    because nothing was able to look for them."""
    items = set(items)
    models = [m for m in detector_models if m["items"] & items]
    if not models: return {}
    H, W = frame.shape[:2]
    ph   = y2 - y1
    pad_top  = max(60, int(ph * 0.25))
    pad_side = max(25, int((x2-x1) * 0.15))
    ex1 = max(0, x1-pad_side); ex2 = min(W, x2+pad_side)
    ey1 = max(0, y1-pad_top)
    # Head/torso is enough (and sharper) for helmet/vest/goggles/smoking;
    # shoes, gloves and harness need the rest of the body in the crop.
    ey2 = min(H, y2 + int(ph*0.05)) if items & FULL_BODY_ITEMS else min(H, y1 + int(ph * 0.65))
    crop = frame[ey1:ey2, ex1:ex2]

    pos = defaultdict(float); neg = defaultdict(float)
    if crop.size > 0:
        for m in models:
            try:
                for r in m["model"](crop, conf=0.30, verbose=False):
                    for box in r.boxes:
                        hit = m["class_map"].get(int(box.cls[0]))
                        if not hit or hit[0] not in items: continue
                        item, positive = hit
                        conf = float(box.conf[0])
                        if positive: pos[item] = max(pos[item], conf)
                        else:        neg[item] = max(neg[item], conf)
            except Exception as e:
                print(f"[DETECTOR ERROR] {m['file']}: {e}")

    covered = set()
    for m in models: covered |= m["items"]
    result = {}
    for item in items & covered:
        if CHECK_ITEMS[item]["kind"] == "behavior":
            result[item] = pos[item] < SMOKING_CONF
        else:
            result[item] = pos[item] >= PPE_CONF and pos[item] >= neg[item]
    return result

# ── PPE WORKER ────────────────────────────────────────────────────────────────
# A check runs every detector that covers the zone's items on a person crop
# (up to ~4 models) - done inline it froze the video for 50-200 ms each
# time. Jobs go to one background thread instead; process_cam_frame picks
# the result up from the track on a later frame. One job per person at a
# time, and the analysis frame is never drawn on, so it can be shared.
_ppe_queue = queue.Queue()
_ppe_worker_started = [False]

def _ppe_worker():
    while True:
        t, frame, box, items = _ppe_queue.get()
        try:
            res = check_items(frame, *box, items)
        except Exception as e:
            print(f"[PPE WORKER ERROR] {e}"); res = {}
        t["ppe_result"] = res

def submit_ppe_check(t, frame, box, items):
    if not _ppe_worker_started[0]:
        threading.Thread(target=_ppe_worker, daemon=True).start()
        _ppe_worker_started[0] = True
    t["ppe_busy"] = True
    _ppe_queue.put((t, frame, box, set(items)))

def violation_short(items):
    """'HELMET & SMOKING' - for on-video labels and the admin log."""
    return " & ".join(CHECK_ITEMS[i]["label"].upper() for i in ordered_checks(items))

def violation_message(items):
    """(title, detail lines) for Telegram / the web dashboard."""
    items = ordered_checks(items)
    ppe = [CHECK_ITEMS[i]["label"].upper() for i in items if CHECK_ITEMS[i]["kind"] == "ppe"]
    smoking = "smoking" in items
    title = ("Smoking Violation" if smoking and not ppe else
             "Safety Violation"  if smoking else "PPE Violation")
    lines = []
    if ppe:     lines.append("Missing: " + " & ".join(ppe))
    if smoking: lines.append("Smoking detected")
    return title, lines

# ── TRACKING ──────────────────────────────────────────────────────────────────
cam_tracked = defaultdict(dict)
cam_nid     = defaultdict(lambda: [0])

# Every camera is analysed at one fixed size, whatever size its tile is
# shown at - so resizing the window or enlarging a camera never changes
# detection, tracking or zone hit-testing (zones scale with the frame).
PROC_W, PROC_H = 960, 720
TRACK_MATCH_PX = 120          # max centre jump (in PROC pixels) to stay the same person

def get_track(ci, cx, cy):
    for tid, d in cam_tracked[ci].items():
        if abs(d["cx"]-cx) < TRACK_MATCH_PX and abs(d["cy"]-cy) < TRACK_MATCH_PX: return tid
    tid = cam_nid[ci][0]; cam_nid[ci][0] += 1
    cam_tracked[ci][tid] = {
        "cx": cx, "cy": cy, "ppe_ok": True, "ppe_confirmed": False,
        "missing": "", "violation_start": None, "grace_active": False,
        "last_ppe": 0, "cached": {}, "trigger_checks": set(),
        "behavior_hist": defaultdict(lambda: deque(maxlen=BEHAVIOR_WINDOW)),
        "confirm_time": 0, "telegram_violation_start": None,
        "violation_alerted": False,
        "triggered": False, "zone_entry_logged": False,
        "rest_entry_start": None, "rest_notified": False}
    return tid

# ── ADMIN LOG — full scrollable history ───────────────────────────────────────
admin_log      = []
log_scroll_pos = [0]
ADMIN_LOG_MAX  = 5000  # caps memory growth on a long-running demo/session

def add_log(text, color=None):
    if color is None: color = THEME["text_primary"]
    now = datetime.now().strftime("%H:%M:%S")
    admin_log.append({"text": f"[{now}] {text}", "color": color})
    if len(admin_log) > ADMIN_LOG_MAX:
        del admin_log[:len(admin_log) - ADMIN_LOG_MAX]
    log_scroll_pos[0] = 0
    print("LOG:", text)

def log_file(event, detail=""):
    try:
        now = datetime.now()
        with open(f"{LOG_PATH}/system_{now.strftime('%Y-%m-%d')}.txt", "a") as f:
            f.write(f"{now.strftime('%H:%M:%S')},{event},{detail}\n")
    except Exception as e:
        print(f"[LOG FILE ERROR] {e}")

# ── HELP CONTENT ──────────────────────────────────────────────────────────────
HELP_CONTENT = [
    ("TOP BAR ICONS", "", ""),
    ("", "Back",    "Return to the setup screen"),
    ("", "Log",     "Admin activity log"),
    ("", "Select",  "Pick zone kind + what to check, then draw"),
    ("", "Tools",   "All zones - click one to edit its checks"),
    ("", "Shift",   "Shift hours + after-hours room alerts"),
    ("", "Power",   "Turn ALL zones off (1-5 h / until on)"),
    ("", "Edit",    "Toggle edit mode (move/resize/delete zones)"),
    ("", "Help",    "This panel"),
    ("", "Pause",   "Pause/resume the selected camera"),
    ("KEYBOARD", "", ""),
    ("", "E",          "Toggle edit mode on/off"),
    ("", "R",          "Reset all person tracking"),
    ("", "B",          "Back to setup screen"),
    ("", "Q",          "Quit the program"),
    ("", "SPACE",      "Pause / resume the selected camera"),
    ("", "S",          "Save a snapshot of the selected camera"),
    ("", "L",          "Selected camera back to live"),
    ("", "+  /  -",    "Resize selected zone"),
    ("", "Delete",     "Delete selected zone"),
    ("", "Up / Down",  "Scroll admin log"),
    ("CAMERAS", "", ""),
    ("", "Click a tile", "Select that camera (highlighted border)"),
    ("", "Double-click", "Enlarge a camera / back to the grid"),
    ("", "F  /  Esc",    "Enlarge selected camera / back to grid"),
    ("REVIEW (RECORDING)", "", ""),
    ("", "Timeline",     "Click / drag to jump in the recording"),
    ("", "Back / Fwd 10","Jump 10 s back / forward"),
    ("", "<< / >>",      "Fast rewind / forward x2-x16"),
    ("", "Pause",        "Freeze picture - detection keeps running"),
    ("", "LIVE",         "Back to the live picture"),
    ("ADMIN LOG", "", ""),
    ("", "Alerts tab",   "Alert photos per day (< > to change day)"),
    ("", "Double-click", "Watch that alert's recording"),
    ("DRAWING ZONES", "", ""),
    ("", "Left-click",  "Add a point to the zone"),
    ("", "Right-click", "Finish & save the zone"),
    ("EDIT MODE", "", ""),
    ("", "Drag dots",    "Move zone points"),
    ("", "Right-click",  "Open zone options menu"),
    ("", "Click segment","Insert a new point"),
    ("ZONE MENU", "", ""),
    ("", "Edit Checks",  "Change what this zone checks / turn it off"),
    ("", "Turn Off / On","Switch a zone off without deleting it"),
    ("", "Zoom In",      "Expand selected zone 10%"),
    ("", "Zoom Out",     "Shrink selected zone 10%"),
    ("", "Delete Zone",  "Remove selected zone"),
    ("", "Exit Program", "Close the monitor"),
]

# ── TOP BAR + DOCKED SIDEBAR ────────────────────────────────────────────────────
APP_NAME       = "SAFETY CAM"
TOP_BAR_H      = 40           # fixed regardless of camera count/window size
PANEL_W        = 240          # fixed docked sidebar width — does not shrink the camera grid
PANEL_HEADER_H = 36
ICON_BTN       = 28
ICON_GAP       = 6

# ── REVIEW BAR (timeline, rewind/forward, speed, LIVE for the selected camera) ─
PLAYBACK_BAR_H      = 68   # timeline row + buttons row
PLAYBACK_SEEK_SECONDS = 10  # how far rewind/forward jump per click

TOP_BAR_TIPS = {
    "back": "Back to setup (B)", "log": "System log", "select": "New zone & checks",
    "tools": "All zones", "shift": "Shift & room watch", "edit": "Edit zones (E)",
    "power": "Turn ALL zones off / on",
    "help": "Help & shortcuts", "pauseplay": "Pause / resume camera (Space)",
}
PLAYBACK_TIPS = {
    "rewind": f"Back {PLAYBACK_SEEK_SECONDS} seconds", "pauseplay": "Pause / play (Space)",
    "forward": f"Forward {PLAYBACK_SEEK_SECONDS} seconds", "live": "Back to live (L)",
    "frew": "Fast rewind  x2 > x4 > x8 > x16", "ffwd": "Fast forward  x2 > x4 > x8 > x16",
    "snap": "Snapshot (S)", "focus": "Enlarge camera (F / double-click)",
}
DOUBLE_CLICK_S = 0.4

def _bar_button(canvas, rects, key, glyph, cx, bar_h, active, hover, half=16):
    cy = bar_h // 2
    if active:
        cv2.rectangle(canvas, (cx-half-2, 2), (cx+half+2, bar_h-2), THEME["accent"], -1)
        col = THEME["white"]
    else:
        if hover == key:
            rounded_rect(canvas, (cx-half, 5), (cx+half, bar_h-5), THEME["border"], -1, r=6)
        col = THEME["text_primary"] if hover == key else THEME["text_secondary"]
    draw_glyph_icon(canvas, glyph, cx, cy, 17, col)
    rects[key] = (cx-half-2, 0, cx+half+2, bar_h)

def draw_top_bar(total_w, active_panel, edit_mode, selected_paused, hover=None, all_paused=False):
    canvas = np.full((TOP_BAR_H, total_w, 3), THEME["bg_header"], dtype=np.uint8)
    cv2.line(canvas, (0, TOP_BAR_H-1), (total_w, TOP_BAR_H-1), THEME["border"], 1)
    rects = {}
    cy = TOP_BAR_H // 2
    step = ICON_BTN + ICON_GAP + 6

    x = 26
    _bar_button(canvas, rects, "back", "back", x, TOP_BAR_H, False, hover)
    x += 26
    cv2.line(canvas, (x, 8), (x, TOP_BAR_H-8), THEME["border"], 1)
    x += 14
    draw_glyph_icon(canvas, "shield", x+9, cy, 18, THEME["accent"])
    put_text(canvas, APP_NAME, (x+24, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME["accent"], 2)
    (tw, _), _ = text_size(APP_NAME, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
    x += 24 + tw + 14
    cv2.line(canvas, (x, 8), (x, TOP_BAR_H-8), THEME["border"], 1)
    x += 12 + ICON_BTN//2

    for key in ("log", "select", "tools", "shift", "power", "edit"):
        active = (active_panel == key) or (key == "edit" and edit_mode)
        _bar_button(canvas, rects, key, key, x, TOP_BAR_H, active, hover)
        if key == "power" and all_paused and not active:      # everything off: say so
            rounded_rect(canvas, (x-14, 6), (x+14, TOP_BAR_H-6), THEME["warn"], -1, r=6)
            draw_glyph_icon(canvas, "power", x, TOP_BAR_H//2, 17, THEME["black"])
        x += step

    rx = total_w - 12 - ICON_BTN//2
    _bar_button(canvas, rects, "help", "help", rx, TOP_BAR_H, active_panel == "help", hover)
    rx -= step
    _bar_button(canvas, rects, "pauseplay", "play" if selected_paused else "pause", rx, TOP_BAR_H,
                selected_paused, hover)
    return canvas, rects

TIMELINE_Y = 15            # centre of the timeline track inside the review bar
REVIEW_SPEEDS = [2, 4, 8, 16]

def _speed_txt(speed):
    if speed == 1: return ""
    return f"{'>>' if speed > 0 else '<<'} x{abs(speed):g}"

def _clock(t, with_day=False):
    d = datetime.fromtimestamp(t)
    txt = d.strftime("%I:%M:%S %p").lstrip("0")
    return (d.strftime("%d %b ") + txt) if with_day else txt

def draw_playback_bar(total_w, selected_paused, selected_name, focus_on=False, hover=None, review=None,
                      spans=None, now=None, rec_on=True):
    """Bottom bar for the selected camera.
    Row 1: timeline of its recording, from the start of the footage to NOW
    (click or drag to jump; recorded stretches lighter, gaps dark).
    Row 2: fast rewind, back 10 s, pause/play, forward 10 s, fast forward,
    LIVE - all of these only change what the tile SHOWS; the live picture
    keeps being analysed. Right: snapshot, enlarge / back to grid.
    Returns (canvas, rects, timeline) - timeline = (t0, t1, x1, x2) or None."""
    canvas = np.full((PLAYBACK_BAR_H, total_w, 3), THEME["bg_header"], dtype=np.uint8)
    cv2.line(canvas, (0, 0), (total_w, 0), THEME["border"], 1)
    rects = {}; timeline = None
    now = now or time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── row 1: timeline ──
    x1, x2 = 150, total_w - 130
    if rec_on and x2 - x1 > 80:
        spans = spans or []
        t1 = now
        t0 = spans[0][0] if spans else now - 60
        t0 = min(t0, t1 - 60)                     # at least a minute wide
        span_t = t1 - t0
        tx = lambda t: int(x1 + (x2 - x1) * min(1.0, max(0.0, (t - t0) / span_t)))
        ty = TIMELINE_Y
        rounded_rect(canvas, (x1, ty-3), (x2, ty+3), THEME["bg_panel"], -1, r=3)
        for a, b in spans:
            xa, xb = tx(a), max(tx(a)+1, tx(b))
            cv2.rectangle(canvas, (xa, ty-3), (xb, ty+3), THEME["accent_dim"], -1)
        if review is not None:
            px = tx(review["t"])
            cv2.rectangle(canvas, (x1, ty-1), (px, ty+1), THEME["warn"], -1)
            cv2.circle(canvas, (px, ty), 7, THEME["warn"], -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, ty), 7, THEME["white"], 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x2, ty), 6, THEME["danger"], -1, cv2.LINE_AA)
        multi_day = datetime.fromtimestamp(t0).date() != datetime.fromtimestamp(t1).date()
        start_lbl = _clock(t0, multi_day) if spans else "no recording yet"
        (sw, _), _ = text_size(start_lbl, font, 0.3, 1)
        put_text(canvas, start_lbl, (x1 - 10 - sw, ty + 4), font, 0.3, THEME["text_secondary"], 1)
        put_text(canvas, "NOW  " + _clock(t1), (x2 + 12, ty + 4), font, 0.3,
                 THEME["ok"] if review is None else THEME["text_secondary"], 1)
        rects["timeline"] = (x1 - 6, 2, x2 + 6, ty + 10)
        timeline = (t0, t1, x1, x2)
    elif not rec_on:
        text_centered(canvas, "Recording is off", total_w // 2, TIMELINE_Y + 4, 0.3, THEME["text_dim"], 1)

    # ── row 2: name + status, transport buttons, LIVE, snapshot / enlarge ──
    cy = TIMELINE_Y + 12 + (PLAYBACK_BAR_H - TIMELINE_Y - 12) // 2
    row = (TIMELINE_Y + 12, PLAYBACK_BAR_H)
    def button(key, glyph, cx, active=False, size=17):
        if active:
            rounded_rect(canvas, (cx-17, row[0]+3), (cx+17, row[1]-4), THEME["accent"], -1, r=6)
            col = THEME["white"]
        else:
            if hover == key:
                rounded_rect(canvas, (cx-17, row[0]+3), (cx+17, row[1]-4), THEME["border"], -1, r=6)
            col = THEME["white"] if hover == key else THEME["text_primary"]
        draw_glyph_icon(canvas, glyph, cx, cy, size, col)
        rects[key] = (cx-18, row[0], cx+18, row[1])

    draw_glyph_icon(canvas, "video", 24, cy, 16, THEME["accent"])
    name = _fit(selected_name, 22)
    put_text(canvas, name, (40, cy+5), font, 0.4, THEME["text_primary"], 1)
    (nw, _), _ = text_size(name, font, 0.4, 1)
    if review is None:
        status, scol = "LIVE", THEME["ok"]
    else:
        status = ("PAUSED  " if review["paused"] else "") + _clock(review["t"])
        sp = "" if review["paused"] else _speed_txt(review.get("speed", 1))
        if sp: status += "   " + sp
        scol = THEME["warn"]
    put_text(canvas, status, (40+nw+14, cy+5), font, 0.36, scol, 1)

    cx = total_w // 2
    speed = review.get("speed", 1) if review else 1
    playing_normal = review is None or (not review["paused"] and speed == 1)
    button("frew", "frew", cx-100, active=review is not None and not review["paused"] and speed < 0, size=21)
    button("rewind", "back10", cx-50, size=25)
    button("pauseplay", "pause" if playing_normal else "play", cx, active=bool(review and review["paused"]), size=19)
    button("forward", "fwd10", cx+50, size=25)
    button("ffwd", "ffwd", cx+100, active=review is not None and not review["paused"] and speed > 1, size=21)
    lx1 = cx + 140; lx2 = lx1 + 66; ly1, ly2 = row[0] + 6, row[1] - 7
    if review is None:
        rounded_rect(canvas, (lx1, ly1), (lx2, ly2), THEME["bg_panel"], -1, r=6)
        cv2.circle(canvas, (lx1+15, cy), 4, THEME["danger"], -1, cv2.LINE_AA)
        put_text(canvas, "LIVE", (lx1+26, cy+5), font, 0.36, THEME["text_secondary"], 1)
    else:
        rounded_rect(canvas, (lx1, ly1), (lx2, ly2), THEME["accent"], -1, r=6)
        if hover == "live": rounded_rect(canvas, (lx1, ly1), (lx2, ly2), THEME["white"], 1, r=6)
        cv2.circle(canvas, (lx1+15, cy), 4, THEME["white"], -1, cv2.LINE_AA)
        put_text(canvas, "LIVE", (lx1+26, cy+5), font, 0.36, THEME["white"], 2)
    rects["live"] = (lx1, row[0], lx2, row[1])

    button("focus", "grid" if focus_on else "expand", total_w-30)
    button("snap", "camera", total_w-72)
    return canvas, rects, timeline

def draw_review_badge(tile, rv, now_ts, compact):
    """Bottom-left pill on a tile that's showing recorded footage."""
    behind = max(0, int(now_ts - rv["t"]))
    m, sec = divmod(behind, 60); hr, m = divmod(m, 60)
    ago = f"-{hr}:{m:02d}:{sec:02d}" if hr else f"-{m}:{sec:02d}"
    label = ("ALERT REVIEW" if rv.get("alert") else "REVIEW")
    label += "  PAUSED" if rv["paused"] else ("  " + _speed_txt(rv.get("speed", 1)) if rv.get("speed", 1) != 1 else "")
    text = label if compact else f"{label}   {ago}"
    fs = 0.3 if compact else 0.38
    (tw, th), _ = text_size(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    H = tile.shape[0]
    y2 = H - 5; y1 = y2 - th - (8 if compact else 12)
    rounded_rect(tile, (5, y1), (5 + tw + 16, y2), THEME["warn"], -1, r=4)
    put_text(tile, text, (13, y2 - (4 if compact else 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, THEME["black"], 1)

def draw_tooltip(img, text, cx, y):
    (tw, th), _ = text_size(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    x1 = int(min(max(4, cx - tw//2 - 8), img.shape[1] - tw - 20)); x2 = x1 + tw + 16; y2 = y + th + 14
    rounded_rect(img, (x1, y), (x2, y2), (52, 48, 48), -1, r=5)
    rounded_rect(img, (x1, y), (x2, y2), THEME["border"], 1, r=5)
    put_text(img, text, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, THEME["text_primary"], 1)

def _panel_shell(w, h, title):
    canvas = np.full((h, w, 3), THEME["bg_panel_alt"], dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (w, PANEL_HEADER_H), THEME["bg_header"], -1)
    put_text(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                THEME["accent"], 1, cv2.LINE_AA)
    cv2.line(canvas, (0, PANEL_HEADER_H), (w, PANEL_HEADER_H), THEME["border"], 1)
    cv2.line(canvas, (0, 0), (0, h), THEME["accent"], 2)
    return canvas

def build_panel_empty(w, h):
    """The sidebar's idle state — fixed width, genuinely empty, no header
    text — filled in only once one of the top-bar icons is clicked."""
    canvas = np.full((h, w, 3), THEME["bg_panel_alt"], dtype=np.uint8)
    cv2.line(canvas, (0, 0), (0, h), THEME["accent"], 2)
    return canvas

TEXT_ENTRY_H = 18
_alert_thumb_cache = {}

def _alert_thumb(day, entry, bw, bh):
    """Letterboxed ('contain') thumbnail of a saved alert photo, cached."""
    key = (day, entry["file"], bw, bh)
    hit = _alert_thumb_cache.get(key)
    if hit is not None: return hit
    box = np.full((bh, bw, 3), THEME["bg_panel"], np.uint8)
    try:
        img = cv2.imdecode(np.fromfile(os.path.join(ALERTS_DIR, day, entry["file"]), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            s_ = min(bw / img.shape[1], bh / img.shape[0])
            nw, nh = max(1, int(img.shape[1]*s_)), max(1, int(img.shape[0]*s_))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            ox, oy = (bw-nw)//2, (bh-nh)//2
            box[oy:oy+nh, ox:ox+nw] = img
    except Exception:
        pass
    if len(_alert_thumb_cache) > 400: _alert_thumb_cache.clear()
    _alert_thumb_cache[key] = box
    return box

def _pretty_day(day):
    if day == datetime.now().strftime("%Y-%m-%d"): return "Today"
    try: return datetime.strptime(day, "%Y-%m-%d").strftime("%a %d %b")
    except ValueError: return day

def build_panel_log(w, h, ui):
    """ui: {"tab": "log"|"alerts", "day", "scroll", "sel"}. Hits are
    (x1,y1,x2,y2,(action,value))."""
    canvas = _panel_shell(w, h, "ADMIN LOG")
    hits = []
    y = PANEL_HEADER_H + 8
    tbw = (w - 24) // 2
    n_today = len(load_alerts(datetime.now().strftime("%Y-%m-%d")))
    for i, (key, label) in enumerate((("log", "EVENTS"), ("alerts", f"ALERTS ({n_today})"))):
        x1 = 10 + i*(tbw+4); x2 = x1 + tbw; on = ui["tab"] == key
        rounded_rect(canvas, (x1, y), (x2, y+24), THEME["accent"] if on else THEME["bg_panel"], -1, r=5)
        if not on: rounded_rect(canvas, (x1, y), (x2, y+24), THEME["border"], 1, r=5)
        text_centered(canvas, label, (x1+x2)//2, y+16, 0.32, THEME["white"] if on else THEME["text_secondary"], 1)
        hits.append((x1, y, x2, y+24, ("tab", key)))
    top = y + 34

    if ui["tab"] == "log":
        avail_h = h - top - 22
        total = len(admin_log); end_idx = total - log_scroll_pos[0]
        visible = admin_log[max(0, end_idx - avail_h // TEXT_ENTRY_H):max(0, end_idx)]
        yy = top
        for e in visible:
            cv2.circle(canvas, (12, yy+6), 3, e["color"], -1)
            put_text(canvas, fit_px(e["text"], w - 30, 0.30), (20, yy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, e["color"], 1)
            yy += TEXT_ENTRY_H
        put_text(canvas, "scroll wheel to browse", (12, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.26, THEME["text_dim"], 1)
        return canvas, hits

    # ── ALERTS tab: one day's alert photos, newest first ──
    days = alert_days()
    if ui["day"] not in days: ui["day"] = days[-1]
    di = days.index(ui["day"])
    for dx, glyph, ok in ((14, "<", di > 0), (w-14, ">", di < len(days)-1)):
        col = THEME["accent"] if ok else THEME["border"]
        text_centered(canvas, glyph, dx, top+12, 0.5, col, 2)
        if ok: hits.append((dx-12, top-4, dx+12, top+20, ("day", -1 if glyph == "<" else 1)))
    entries = load_alerts(ui["day"])
    text_centered(canvas, f"{_pretty_day(ui['day'])}  -  {len(entries)} alert{'s' if len(entries) != 1 else ''}",
                  w//2, top+12, 0.34, THEME["text_primary"], 1)
    top += 26
    cols_ = 2; gap = 6; cw = (w - 20 - gap) // cols_; th_ = int(cw * 0.95); ch = th_ + 18
    rows_fit = max(1, (h - top - 22) // (ch + gap))
    order = list(reversed(entries))
    max_scroll = max(0, (len(order) + cols_ - 1) // cols_ - rows_fit)
    ui["scroll"] = min(ui["scroll"], max_scroll)
    start = ui["scroll"] * cols_
    for k, entry in enumerate(order[start:start + rows_fit*cols_]):
        r_, c_ = divmod(k, cols_)
        x1 = 10 + c_*(cw+gap); y1 = top + r_*(ch+gap)
        canvas[y1:y1+th_, x1:x1+cw] = _alert_thumb(ui["day"], entry, cw, th_)
        sel = ui.get("sel") is entry
        cv2.rectangle(canvas, (x1, y1), (x1+cw-1, y1+th_-1), THEME["accent"] if sel else THEME["danger"], 2 if sel else 1)
        cap = datetime.fromtimestamp(entry["ts"]).strftime("%I:%M:%S %p").lstrip("0") + f"  CAM{entry['cam']+1}"
        put_text(canvas, fit_px(cap, cw, 0.28), (x1+1, y1+th_+13), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                 THEME["text_primary"] if sel else THEME["text_secondary"], 1)
        hits.append((x1, y1, x1+cw, y1+ch, ("alert", entry)))
    if not entries:
        text_centered(canvas, "No alerts on this day", w//2, top+30, 0.34, THEME["text_dim"], 1)
    put_text(canvas, "double-click a photo to watch it", (12, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
             THEME["text_dim"], 1)
    return canvas, hits

def _fit(text, max_chars):
    return text if len(text) <= max_chars else text[:max_chars-2] + ".."

_fit_px_cache = {}
def fit_px(text, max_w, scale, thickness=1):
    """Cuts text to fit max_w pixels in the UI font (cached - panels redraw
    every frame)."""
    key = (text, max_w, scale, thickness)
    hit = _fit_px_cache.get(key)
    if hit is None:
        hit = text
        if text_size(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_w:
            while hit and text_size(hit + "..", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_w:
                hit = hit[:-1]
            hit = hit.rstrip() + ".."
        if len(_fit_px_cache) > 4000: _fit_px_cache.clear()
        _fit_px_cache[key] = hit
    return hit

def build_panel_select(w, h, kind, checks, editing_label, supported, off_zone=None):
    """Zone builder. In "new zone" mode it sets up what the next drawn zone
    will be; when editing_label is set, the same controls change an
    existing zone live instead (opened from the Tools list or the zone's
    right-click menu). Hits are (x1,y1,x2,y2,(action,value))."""
    canvas = _panel_shell(w, h, "EDIT ZONE" if editing_label else "NEW ZONE")
    hits = []
    y = PANEL_HEADER_H + 10
    if editing_label:
        put_text(canvas, _fit(editing_label, 34), (12, y+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, THEME["text_secondary"], 1, cv2.LINE_AA)
        y += 18

    put_text(canvas, "ZONE KIND", (12, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                THEME["text_dim"], 1, cv2.LINE_AA)
    y += 14
    for ztype, zinfo in ZONE_TYPES.items():
        col = zinfo["color"]; is_sel = (kind == ztype)
        bx1, by1, bx2, by2 = 10, y, w-10, y+23
        rounded_rect(canvas, (bx1,by1), (bx2,by2), col if is_sel else THEME["bg_panel"], -1, r=6)
        rounded_rect(canvas, (bx1,by1), (bx2,by2), col, 1, r=6)
        tc = THEME["black"] if is_sel else col
        zinfo["icon"](canvas, bx1+16, (by1+by2)//2, 8, tc, 2)
        put_text(canvas, zinfo["label"], (bx1+32, (by1+by2)//2+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, tc, 1, cv2.LINE_AA)
        hits.append((bx1, by1, bx2, by2, ("kind", ztype)))
        y += 27

    has_checks = ZONE_TYPES[kind]["has_checks"]
    if has_checks:
        y += 6
        put_text(canvas, "CHECK FOR", (12, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    THEME["text_dim"], 1, cv2.LINE_AA)
        y += 14
        for item, info in CHECK_ITEMS.items():
            on = item in checks
            rx1, ry1, rx2, ry2 = 10, y, w-10, y+20
            bx, by = 16, y+4
            cv2.rectangle(canvas, (bx, by), (bx+12, by+12),
                          THEME["accent"] if on else THEME["border"], -1 if on else 1)
            if on:
                cv2.line(canvas, (bx+3, by+6), (bx+5, by+9), THEME["white"], 2)
                cv2.line(canvas, (bx+5, by+9), (bx+10, by+3), THEME["white"], 2)
            put_text(canvas, info["label"], (36, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        THEME["text_primary"] if on else THEME["text_secondary"], 1, cv2.LINE_AA)
            if item not in supported:
                (tw, _), _ = text_size("no model", cv2.FONT_HERSHEY_SIMPLEX, 0.26, 1)
                put_text(canvas, "no model", (w-14-tw, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                            THEME["warn"], 1, cv2.LINE_AA)
            hits.append((rx1, ry1, rx2, ry2, ("item", item)))
            y += 20

    if editing_label and off_zone is not None:
        y += 8
        put_text(canvas, "ZONE STATUS", (12, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                 THEME["text_dim"], 1, cv2.LINE_AA)
        y += 14
        if zones_paused():
            put_text(canvas, "ALL zones are off " + zones_pause_text(), (14, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                     THEME["warn"], 1, cv2.LINE_AA)
            put_text(canvas, "(power button in the top bar)", (14, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                     THEME["text_dim"], 1, cv2.LINE_AA)
            y += 36
        elif zone_on(off_zone):
            put_text(canvas, "ON - turn off for:", (14, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                     THEME["ok"], 1, cv2.LINE_AA)
            y += 22
            k = len(ZONE_OFF_HOURS); bw = (w - 20 - (k-1)*4) // k
            for i, hrs in enumerate(ZONE_OFF_HOURS):
                bx1 = 10 + i*(bw+4)
                rounded_rect(canvas, (bx1, y), (bx1+bw, y+24), THEME["bg_panel"], -1, r=5)
                rounded_rect(canvas, (bx1, y), (bx1+bw, y+24), THEME["warn"], 1, r=5)
                text_centered(canvas, f"{hrs}h", bx1+bw//2, y+16, 0.34, THEME["warn"], 1)
                hits.append((bx1, y, bx1+bw, y+24, ("off", hrs)))
            y += 28
            rounded_rect(canvas, (10, y), (w-10, y+24), THEME["bg_panel"], -1, r=5)
            rounded_rect(canvas, (10, y), (w-10, y+24), THEME["warn"], 1, r=5)
            text_centered(canvas, "OFF UNTIL I TURN IT ON", w//2, y+16, 0.32, THEME["warn"], 1)
            hits.append((10, y, w-10, y+24, ("off", "manual")))
            y += 26
        else:
            put_text(canvas, zone_off_text(off_zone), (14, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                     THEME["warn"], 1, cv2.LINE_AA)
            y += 22
            rounded_rect(canvas, (10, y), (w-10, y+24), THEME["ok"], -1, r=5)
            text_centered(canvas, "TURN ON NOW", w//2, y+16, 0.34, THEME["black"], 1)
            hits.append((10, y, w-10, y+24, ("on", None)))
            y += 26

    y += 10
    bx1, by1, bx2, by2 = 10, y, w-10, y+30
    if editing_label:
        rounded_rect(canvas, (bx1,by1), (bx2,by2), THEME["accent"], -1, r=6)
        text_centered(canvas, "DONE", (bx1+bx2)//2, by1+20, 0.42, THEME["white"], 1)
        hits.append((bx1, by1, bx2, by2, ("done", None)))
    else:
        ready = (not has_checks) or bool(checks)
        rounded_rect(canvas, (bx1,by1), (bx2,by2), THEME["ok"] if ready else THEME["bg_panel"], -1, r=6)
        if not ready: rounded_rect(canvas, (bx1,by1), (bx2,by2), THEME["border"], 1, r=6)
        text_centered(canvas, "DRAW ZONE", (bx1+bx2)//2, by1+20, 0.42,
                      THEME["black"] if ready else THEME["text_dim"], 1)
        if ready: hits.append((bx1, by1, bx2, by2, ("draw", None)))
        y += 40
        hint = ("Click points on a camera," if ready else "Tick at least one check")
        put_text(canvas, hint, (12, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, THEME["text_dim"], 1)
        if ready:
            put_text(canvas, "right-click to finish.", (12, y+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, THEME["text_dim"], 1)
    return canvas, hits

def build_panel_tools(w, h, zone_editor, cameras, supported):
    canvas = _panel_shell(w, h, "ZONES")
    y = PANEL_HEADER_H + 14
    put_text(canvas, "Click a zone to edit its checks", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, THEME["text_dim"], 1)
    y += 14
    hits = []; total = 0; unsupported = set()
    for ci, cam in enumerate(cameras):
        zones = zone_editor.get_zones(ci)
        if not zones: continue
        if y > h - 30: break
        put_text(canvas, _fit(cam["name"].upper(), 30), (12, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, THEME["accent"], 1, cv2.LINE_AA)
        y += 16
        for zi, zone in enumerate(zones):
            if y > h - 30: break
            total += 1
            zinfo = ZONE_TYPES[zone["type"]]
            is_sel = (zone_editor.selected_index == ci and zone_editor.selected_zone == zi)
            if is_sel:
                cv2.rectangle(canvas, (4, y), (w-4, y+28), THEME["accent_dim"], -1)
            zinfo["icon"](canvas, 20, y+14, 7, zinfo["color"], 1)
            put_text(canvas, zinfo["label"], (34, y+11), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        zinfo["color"], 1, cv2.LINE_AA)
            if not zone_on(zone):
                put_text(canvas, _fit(zone_off_text(zone), 34), (34, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                         THEME["warn"], 1, cv2.LINE_AA)
            elif zinfo["has_checks"]:
                checks = zone.get("checks", [])
                unsupported |= set(checks) - supported
                detail = ", ".join(CHECK_ITEMS[c]["label"] for c in checks) or "no checks"
                put_text(canvas, _fit(detail, 34), (34, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                            THEME["text_secondary"], 1, cv2.LINE_AA)
            hits.append((4, y, w-4, y+28, (ci, zi)))
            y += 32
    if not total:
        put_text(canvas, "No zones drawn yet", (12, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, THEME["text_dim"], 1)
    if unsupported:
        msg = "No model yet: " + ", ".join(CHECK_ITEMS[c]["label"] for c in ordered_checks(unsupported))
        put_text(canvas, _fit(msg, 40), (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                    THEME["warn"], 1, cv2.LINE_AA)
    return canvas, hits

def build_panel_power(w, h, zone_editor):
    """Turn ALL zones off for 1-5 hours or until turned back on, or back on.
    Hits are (x1,y1,x2,y2,(action,value))."""
    canvas = _panel_shell(w, h, "ALL ZONES ON / OFF")
    hits = []; font = cv2.FONT_HERSHEY_SIMPLEX
    total = sum(len(z) for z in zone_editor.zones.values())
    y = PANEL_HEADER_H + 14
    if zones_paused():
        put_text(canvas, "ALL ZONES ARE OFF", (12, y+12), font, 0.42, THEME["warn"], 2)
        put_text(canvas, zones_pause_text(), (12, y+32), font, 0.34, THEME["warn"], 1)
        y += 50
        rounded_rect(canvas, (10, y), (w-10, y+32), THEME["ok"], -1, r=6)
        text_centered(canvas, "TURN ALL ZONES ON NOW", w//2, y+21, 0.38, THEME["black"], 1)
        hits.append((10, y, w-10, y+32, ("resume", None)))
        y += 44
    else:
        own_off = sum(1 for zs in zone_editor.zones.values() for z in zs if not zone_on(z))
        put_text(canvas, f"{total - own_off} of {total} zones are ON", (12, y+12), font, 0.4, THEME["ok"], 1)
        y += 30
        put_text(canvas, "TURN ALL OFF FOR", (12, y+8), font, 0.28, THEME["text_dim"], 1)
        y += 14
        k = len(ZONE_OFF_HOURS); bw = (w - 20 - (k-1)*4) // k
        for i, hrs in enumerate(ZONE_OFF_HOURS):
            bx1 = 10 + i*(bw+4)
            rounded_rect(canvas, (bx1, y), (bx1+bw, y+28), THEME["bg_panel"], -1, r=5)
            rounded_rect(canvas, (bx1, y), (bx1+bw, y+28), THEME["warn"], 1, r=5)
            text_centered(canvas, f"{hrs}h", bx1+bw//2, y+19, 0.38, THEME["warn"], 1)
            hits.append((bx1, y, bx1+bw, y+28, ("pause", hrs)))
        y += 36
        rounded_rect(canvas, (10, y), (w-10, y+30), THEME["warn"], -1, r=6)
        text_centered(canvas, "OFF UNTIL I TURN THEM ON", w//2, y+20, 0.36, THEME["black"], 1)
        hits.append((10, y, w-10, y+30, ("pause", "manual")))
        y += 42
    for line in ("Each zone keeps its own on/off", "setting - it comes back as it was.",
                 "Recording and the after-hours", "room watch keep working."):
        put_text(canvas, line, (12, y+10), font, 0.28, THEME["text_dim"], 1); y += 15
    return canvas, hits

def build_panel_help(w, h):
    canvas = _panel_shell(w, h, "SHORTCUTS & HELP")
    y = PANEL_HEADER_H + 18
    for (cat, key, desc) in HELP_CONTENT:
        if y > h - 12: break
        if cat:
            if y > PANEL_HEADER_H + 18:
                y += 6
                cv2.line(canvas, (10, y-9), (w-10, y-9), THEME["border"], 1)
            put_text(canvas, cat, (12, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        THEME["accent"], 1, cv2.LINE_AA)
            y += 15
        else:
            (kw, _), _ = text_size(key, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
            bx1 = 12; bx2 = 12+max(kw+8, 44)
            rounded_rect(canvas, (bx1, y-2), (bx2, y+11), THEME["bg_panel"], -1, r=3)
            rounded_rect(canvas, (bx1, y-2), (bx2, y+11), THEME["border"], 1, r=3)
            put_text(canvas, key, (bx1+4, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                        THEME["text_primary"], 1, cv2.LINE_AA)
            wrapped = fit_px(desc, w - (bx2+6) - 8, 0.26)
            put_text(canvas, wrapped, (bx2+6, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                        THEME["text_secondary"], 1, cv2.LINE_AA)
            y += 16
    return canvas, []

# ── FLOATING CONTEXT MENU ─────────────────────────────────────────────────────
ZONE_MENU_ITEMS = [
    ("Edit Checks", "accent"),
    ("Turn Off / On", "accent"),
    ("Zoom In",     "accent"),
    ("Zoom Out",    "accent"),
    ("Delete Zone", "danger"),
    ("Exit Program","text_secondary"),
]
MENU_W = 168; MENU_ITEM_H = 30; MENU_PAD = 8

def draw_zone_menu(img, x1, y1):
    h  = MENU_PAD * 2 + MENU_ITEM_H * len(ZONE_MENU_ITEMS)
    x2 = x1 + MENU_W; y2 = y1 + h
    cv2.rectangle(img, (x1+3, y1+3), (x2+3, y2+3), (0,0,0), -1)
    rounded_rect(img, (x1, y1), (x2, y2), THEME["bg_panel_alt"], -1, r=10)
    rounded_rect(img, (x1, y1), (x2, y2), THEME["accent"], 1, r=10)
    for i, (label, _) in enumerate(ZONE_MENU_ITEMS):
        iy = y1 + MENU_PAD + i * MENU_ITEM_H
        if label == "Exit Program":
            cv2.line(img, (x1+10, iy), (x2-10, iy), THEME["border"], 1)
        col = (THEME["danger"]         if label == "Delete Zone"  else
               THEME["text_secondary"] if label == "Exit Program" else
               THEME["accent"])
        put_text(img, label, (x1+16, iy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
    return x1, y1, x2, y2

def zone_menu_hit(x1, y1, x, y):
    rel_y = y - (y1 + MENU_PAD)
    if rel_y < 0: return None
    idx = rel_y // MENU_ITEM_H
    return idx if 0 <= idx < len(ZONE_MENU_ITEMS) else None

def get_edit_badge_rect(PW):
    x2 = PW-6; x1 = x2-40
    return (x1, 4, x2, 24)

def draw_edit_badge(frame):
    PW = frame.shape[1]
    x1, y1, x2, y2 = get_edit_badge_rect(PW)
    rounded_rect(frame,(x1,y1),(x2,y2),THEME["accent_dim"],-1,r=5)
    icon_edit(frame, x1+10, (y1+y2)//2, 5, THEME["accent"], 1)
    put_text(frame,"EDIT",(x1+17,y2-6),
                cv2.FONT_HERSHEY_SIMPLEX,0.34,THEME["accent"],1,cv2.LINE_AA)

def draw_model_warning(frame, missing_items):
    if not missing_items: return
    PW = frame.shape[1]
    msg = "NO MODEL YET: " + ", ".join(CHECK_ITEMS[i]["label"] for i in ordered_checks(missing_items))
    # Just under the camera-name header (drawn later at y 0-28), so it never
    # sits under the bottom-right timestamp badge.
    y1, y2 = 28, 50
    cv2.rectangle(frame,(0,y1),(PW,y2),(20,20,60),-1)
    cv2.rectangle(frame,(0,y1),(PW,y2),THEME["warn"],1)
    icon_alert(frame, 14, (y1+y2)//2, 6, THEME["warn"], 1)
    put_text(frame,msg,(26,y2-7),cv2.FONT_HERSHEY_SIMPLEX,0.38,THEME["warn"],1)

# ── SHIFT & AFTER-HOURS ROOM WATCH ────────────────────────────────────────────
# Outside shift hours (after the end + a short delay for people to leave)
# each watched camera's room should be dark and empty. Lights are judged
# from the frame's own brightness - no extra model needed - and people from
# the person detector that already runs for PPE. What counts as a problem
# (lights, people, or either) is chosen live from the Shift panel.
ROOM_MODES = [("lights", "Lights on"), ("people", "People inside"), ("both", "Lights or people")]
ROOM_SAMPLE_EVERY  = 1.0    # s between brightness samples per camera
ROOM_SAMPLES       = 10     # brightness is averaged over this many samples
ROOM_PEOPLE_MEMORY = 3.0    # s a person sighting counts (detector flickers)
ROOM_ALERT_HOLD    = 15.0   # s a problem must last before it alerts
ROOM_REALERT       = 30*60  # s before the same ongoing problem alerts again
ROOM_DEFAULT_LUX   = 60     # mean gray level for "lights on" until calibrated
ROOM_IR_SAT_MAX    = 3.0    # colourless picture = camera's IR night mode = lights off
SHIFT_STEP_MIN     = 15

def default_after_hours():
    return {"enabled": False, "shift_start": "08:00", "shift_end": "16:00",
            "grace_min": 10, "mode": "both", "off_cams": [], "calib": {}}

def _hm_to_min(s):
    try:
        h, m = str(s).split(":"); return (int(h) % 24) * 60 + int(m) % 60
    except Exception:
        return 0

def _min_to_hm(m):
    m %= 1440; return f"{m//60:02d}:{m%60:02d}"

def fmt_12h(hm):
    m = _hm_to_min(hm); h = m // 60
    return f"{(h % 12) or 12}:{m % 60:02d} {'AM' if h < 12 else 'PM'}"

class RoomWatch:
    def __init__(self, cfg):
        self.cfg = cfg
        ah = cfg.setdefault("after_hours", {})
        for k, v in default_after_hours().items(): ah.setdefault(k, v)
        self.samples     = defaultdict(lambda: deque(maxlen=ROOM_SAMPLES))
        self.last_sample = defaultdict(float)
        self.last_person = defaultdict(float)
        self.people_now  = defaultdict(int)
        self.cond_since  = {}
        self.last_alert  = {}       # ci -> (ts, reasons)
        self.status      = {}       # ci -> dict for the panel / tile badge
        self.simulate    = False    # "test now" - pretend it's after hours
        self.was_after   = None

    @property
    def s(self): return self.cfg["after_hours"]

    def save(self): save_config(self.cfg)

    def is_after_hours(self, now=None):
        if self.simulate: return True
        now = now or datetime.now()
        m = now.hour * 60 + now.minute
        st, en = _hm_to_min(self.s["shift_start"]), _hm_to_min(self.s["shift_end"])
        if st == en: return False                       # 24h shift
        in_shift = (st <= m < en) if st < en else (m >= st or m < en)
        in_grace = (m - en) % 1440 < int(self.s.get("grace_min", 0))
        return not in_shift and not in_grace

    def watched(self, ci):
        return self.s["enabled"] and ci not in self.s.get("off_cams", [])

    def threshold(self, ci):
        c = self.s.get("calib", {}).get(str(ci), {})
        on, off = c.get("on"), c.get("off")
        if on is not None and off is not None: return (on + off) / 2
        if on is not None:  return on * 0.6
        if off is not None: return off + 25
        return ROOM_DEFAULT_LUX

    def measure(self, ci, frame, now_ts):
        """Call with the raw camera frame, before any overlay is drawn on it."""
        if now_ts - self.last_sample[ci] < ROOM_SAMPLE_EVERY: return
        self.last_sample[ci] = now_ts
        small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
        gray = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        sat  = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        self.samples[ci].append((gray, sat))

    def brightness(self, ci):
        s = self.samples[ci]
        if not s: return None, None
        return sum(g for g, _ in s) / len(s), sum(v for _, v in s) / len(s)

    def lights_on(self, ci):
        g, sat = self.brightness(ci)
        if g is None: return None
        if sat < ROOM_IR_SAT_MAX: return False
        return g >= self.threshold(ci)

    def calibrate(self, ci, which):
        calib = self.s.setdefault("calib", {})
        if which == "reset":
            calib.pop(str(ci), None); self.save(); return True
        g, _ = self.brightness(ci)
        if g is None: return False
        calib.setdefault(str(ci), {})[which] = round(g, 1)
        self.save(); return True

    def tick(self):
        """Once per loop: logs shift start/end transitions."""
        if not self.s["enabled"]:
            self.was_after = None; return
        after = self.is_after_hours()
        if after != self.was_after and (after or self.was_after is not None):
            if after:
                add_log("Shift over - after-hours room watch active", THEME["warn"])
                log_file("SHIFT_END", self.s["shift_end"])
            else:
                add_log("Shift started - room watch paused", THEME["ok"])
                log_file("SHIFT_START", self.s["shift_start"])
        self.was_after = after

    def update(self, ci, frame, n_people, now_ts, cam_name):
        """Call after person detection ran on this frame; frame is the photo
        sent with an alert. Returns the status drawn on the tile."""
        if n_people: self.last_person[ci] = now_ts; self.people_now[ci] = n_people
        elif now_ts - self.last_person[ci] > ROOM_PEOPLE_MEMORY: self.people_now[ci] = 0
        lit = self.lights_on(ci); people = self.people_now[ci]
        after = self.is_after_hours()
        st = {"watched": self.watched(ci), "after": after, "lit": lit, "people": people,
              "reasons": [], "state": "idle"}
        self.status[ci] = st
        if not st["watched"] or not after:
            self.cond_since.pop(ci, None); self.last_alert.pop(ci, None); return st
        mode = self.s.get("mode", "both")
        if mode in ("lights", "both") and lit: st["reasons"].append("lights")
        if mode in ("people", "both") and people: st["reasons"].append("people")
        if not st["reasons"]:
            if ci in self.last_alert:
                add_log(f"{cam_name}: room clear now", THEME["ok"])
                log_file("ROOM_CLEAR", f"cam{ci+1}")
            self.cond_since.pop(ci, None); self.last_alert.pop(ci, None)
            st["state"] = "clear"; return st
        since = self.cond_since.setdefault(ci, now_ts)
        if now_ts - since < ROOM_ALERT_HOLD:
            st["state"] = "pending"; return st
        st["state"] = "alarm"
        last = self.last_alert.get(ci)
        if last is None or now_ts - last[0] >= ROOM_REALERT or not set(st["reasons"]) <= last[1]:
            self.last_alert[ci] = (now_ts, set(st["reasons"]))
            lines = []
            if "lights" in st["reasons"]: lines.append("Lights are ON")
            if "people" in st["reasons"]:
                lines.append(f"{people} {'person' if people == 1 else 'people'} in the room")
            when = "Test mode" if self.simulate else f"Shift ended at {fmt_12h(self.s['shift_end'])}"
            msg = (f"After-hours alert CAM{ci+1} ({cam_name})!\n" + "\n".join(lines) +
                   f"\n{when}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            send_telegram_photo(msg, frame, cooldown=0)   # RoomWatch rate-limits itself
            record_web_event(msg, frame.copy(), THEME["warn"], cam=ci)
            log_file("AFTER_HOURS", f"cam{ci+1},{'+'.join(st['reasons'])},{people}")
        return st

def draw_room_badge(frame, st):
    """Bottom-left of a tile, only while its room is being watched after
    hours: state pill + lights + people (the timestamp sits bottom-right)."""
    if not st or not st["watched"] or not st["after"]: return
    PH = frame.shape[0]
    y1, y2 = PH - 28, PH - 6
    state_col = {"alarm": THEME["danger"], "pending": THEME["warn"], "clear": THEME["ok"]}.get(
        st["state"], THEME["text_dim"])
    label = {"alarm": "AFTER HOURS!", "pending": "CHECKING", "clear": "ROOM CLEAR"}.get(st["state"], "AFTER HOURS")
    (tw, _), _ = text_size(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    lit_txt = "--" if st["lit"] is None else ("ON" if st["lit"] else "OFF")
    ppl_txt = str(st["people"])
    (lw, _), _ = text_size(lit_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    (pw_, _), _ = text_size(ppl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    x2 = 6 + tw + 14 + 22 + lw + 12 + 22 + pw_ + 10
    rounded_rect(frame, (6, y1), (x2, y2), THEME["bg_header"], -1, r=4)
    rounded_rect(frame, (6, y1), (6 + tw + 14, y2), state_col, -1, r=4)
    put_text(frame, label, (13, y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, THEME["black"], 1, cv2.LINE_AA)
    x = 6 + tw + 14 + 4
    lc = THEME["danger"] if "lights" in st["reasons"] else (THEME["warn"] if st["lit"] else THEME["text_secondary"])
    draw_glyph_icon(frame, "bulb", x + 9, (y1 + y2) // 2, 16, lc)
    put_text(frame, lit_txt, (x + 20, y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc, 1, cv2.LINE_AA)
    x += 22 + lw + 12
    pc = THEME["danger"] if "people" in st["reasons"] else (THEME["warn"] if st["people"] else THEME["text_secondary"])
    draw_glyph_icon(frame, "people", x + 9, (y1 + y2) // 2, 16, pc)
    put_text(frame, ppl_txt, (x + 20, y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pc, 1, cv2.LINE_AA)

def build_panel_shift(w, h, rw, cameras, selected_cam):
    """Shift hours + after-hours room watch settings, all live. Hits are
    (x1,y1,x2,y2,(action,value))."""
    canvas = _panel_shell(w, h, "SHIFT & ROOM WATCH")
    s = rw.s; hits = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    def checkbox(y, on, label, payload, right=None, right_col=None):
        bx, by = 16, y + 4
        cv2.rectangle(canvas, (bx, by), (bx+12, by+12), THEME["accent"] if on else THEME["border"], -1 if on else 1)
        if on:
            cv2.line(canvas, (bx+3, by+6), (bx+5, by+9), THEME["white"], 2)
            cv2.line(canvas, (bx+5, by+9), (bx+10, by+3), THEME["white"], 2)
        put_text(canvas, label, (36, y+14), font, 0.36,
                    THEME["text_primary"] if on else THEME["text_secondary"], 1, cv2.LINE_AA)
        if right:
            (tw, _), _ = text_size(right, font, 0.28, 1)
            put_text(canvas, right, (w-14-tw, y+14), font, 0.28, right_col or THEME["text_dim"], 1, cv2.LINE_AA)
        hits.append((10, y, w-10, y+20, payload))
    def section(y, title):
        put_text(canvas, title, (12, y+8), font, 0.28, THEME["text_dim"], 1, cv2.LINE_AA)
    def stepper(y, label, value, payload_minus, payload_plus):
        put_text(canvas, label, (14, y+15), font, 0.34, THEME["text_secondary"], 1, cv2.LINE_AA)
        for bx1, txt, pl in ((w-118, "-", payload_minus), (w-34, "+", payload_plus)):
            rounded_rect(canvas, (bx1, y), (bx1+24, y+22), THEME["bg_panel"], -1, r=4)
            rounded_rect(canvas, (bx1, y), (bx1+24, y+22), THEME["accent"], 1, r=4)
            text_centered(canvas, txt, bx1+12, y+16, 0.5, THEME["accent"], 1)
            hits.append((bx1-2, y-2, bx1+26, y+24, pl))
        text_centered(canvas, value, w-64, y+15, 0.36, THEME["text_primary"], 1)

    y = PANEL_HEADER_H + 8
    checkbox(y, s["enabled"], "After-hours alerts", ("toggle", None),
             "ON" if s["enabled"] else "OFF", THEME["ok"] if s["enabled"] else THEME["text_dim"])
    y += 28
    section(y, "SHIFT HOURS"); y += 14
    stepper(y, "Start", fmt_12h(s["shift_start"]), ("start", -SHIFT_STEP_MIN), ("start", SHIFT_STEP_MIN)); y += 28
    stepper(y, "End",   fmt_12h(s["shift_end"]),   ("end", -SHIFT_STEP_MIN),   ("end", SHIFT_STEP_MIN));   y += 28
    stepper(y, "Delay", f"{int(s.get('grace_min', 0))} min", ("grace", -5), ("grace", 5)); y += 30

    section(y, "AFTER HOURS, ALERT IF"); y += 14
    for key, label in ROOM_MODES:
        on = s.get("mode", "both") == key
        cx, cy = 22, y + 10
        cv2.circle(canvas, (cx, cy), 6, THEME["accent"] if on else THEME["border"], 1, cv2.LINE_AA)
        if on: cv2.circle(canvas, (cx, cy), 3, THEME["accent"], -1, cv2.LINE_AA)
        put_text(canvas, label, (36, y+14), font, 0.36,
                    THEME["text_primary"] if on else THEME["text_secondary"], 1, cv2.LINE_AA)
        hits.append((10, y, w-10, y+20, ("mode", key)))
        y += 22
    y += 6

    section(y, "ROOMS (CAMERAS)"); y += 14
    for ci, cam in enumerate(cameras):
        if y > h - 140: break
        st = rw.status.get(ci, {}); lit = rw.lights_on(ci)
        g, _ = rw.brightness(ci)
        info = ("light --" if lit is None else f"light {'ON' if lit else 'off'} {g:.0f}") + f" | {st.get('people', 0)}p"
        checkbox(y, ci not in s.get("off_cams", []), _fit(cam["name"], 12), ("cam", ci), info,
                 THEME["warn"] if lit or st.get("people") else THEME["text_dim"])
        y += 22
    y += 6

    if 0 <= selected_cam < len(cameras) and y < h - 110:
        section(y, _fit(f"LIGHT LEVEL: {cameras[selected_cam]['name'].upper()}", 34)); y += 14
        c = s.get("calib", {}).get(str(selected_cam), {})
        bw = (w - 26) // 2
        for i, (which, txt) in enumerate((("on", "LIGHTS ON NOW"), ("off", "LIGHTS OFF NOW"))):
            bx1 = 10 + i * (bw + 6)
            done = which in c
            rounded_rect(canvas, (bx1, y), (bx1+bw, y+24), THEME["accent_dim"] if done else THEME["bg_panel"], -1, r=5)
            rounded_rect(canvas, (bx1, y), (bx1+bw, y+24), THEME["accent"], 1, r=5)
            text_centered(canvas, txt, bx1+bw//2, y+16, 0.30, THEME["text_primary"], 1)
            hits.append((bx1, y, bx1+bw, y+24, ("calib", which)))
        y += 28
        thr = rw.threshold(selected_cam)
        hint = f"threshold {thr:.0f} ({'learned' if c else 'default'})"
        put_text(canvas, hint, (12, y+10), font, 0.28, THEME["text_dim"], 1, cv2.LINE_AA)
        if c:
            (tw, _), _ = text_size("reset", font, 0.28, 1)
            put_text(canvas, "reset", (w-14-tw, y+10), font, 0.28, THEME["accent"], 1, cv2.LINE_AA)
            hits.append((w-20-tw, y-2, w-8, y+14, ("calib", "reset")))

    # status line + test button pinned to the bottom
    by2 = h - 10; by1 = by2 - 28
    if not s["enabled"]:
        now_txt, now_col = "Room watch is OFF", THEME["text_dim"]
    elif rw.simulate:
        now_txt, now_col = "TEST: acting as after hours", THEME["warn"]
    elif rw.is_after_hours():
        now_txt, now_col = f"AFTER HOURS until {fmt_12h(s['shift_start'])}", THEME["warn"]
    else:
        end_g = _min_to_hm(_hm_to_min(s["shift_end"]) + int(s.get("grace_min", 0)))
        now_txt, now_col = f"In shift - watch from {fmt_12h(end_g)}", THEME["ok"]
    put_text(canvas, now_txt, (12, by1-8), font, 0.32, now_col, 1, cv2.LINE_AA)
    rounded_rect(canvas, (10, by1), (w-10, by2), THEME["warn"] if rw.simulate else THEME["bg_panel"], -1, r=6)
    rounded_rect(canvas, (10, by1), (w-10, by2), THEME["warn"], 1, r=6)
    text_centered(canvas, "STOP TEST" if rw.simulate else "TEST AFTER-HOURS NOW", w//2, by1+19, 0.38,
                  THEME["black"] if rw.simulate else THEME["warn"], 1)
    hits.append((10, by1, w-10, by2, ("test", None)))
    return canvas, hits

# ── PROCESS FRAME ─────────────────────────────────────────────────────────────
class PersonDetector:
    """Runs the person model on its own thread so the video never waits for
    it: the main loop hands over each camera's newest frame, the thread
    works through cameras oldest-first and keeps the latest result per
    camera (boxes + the exact frame they were found in, so PPE crops line
    up with them). On a fast PC that's every frame; on a weak laptop it
    just updates less often while the picture stays smooth."""
    def __init__(self, model, conf=0.45):
        self.model = model; self.conf = conf
        self.cv = threading.Condition()
        self.pending = {}          # ci -> (frame, ts) waiting to be analysed
        self.results = {}          # ci -> (boxes, frame, ts, full-resolution frame)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, ci, frame, ts, raw=None):
        with self.cv:
            self.pending[ci] = (frame, ts, raw); self.cv.notify()

    def get(self, ci):
        with self.cv: return self.results.get(ci)

    def stop(self):
        with self.cv:
            self.running = False; self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while self.running and not self.pending: self.cv.wait()
                if not self.running: return
                ci = min(self.pending, key=lambda c: self.pending[c][1])   # oldest frame first
                frame, ts, raw = self.pending.pop(ci)
            boxes = []
            try:
                for r in self.model(frame, conf=self.conf):
                    for box in r.boxes:
                        if self.model.names[int(box.cls[0])] == "person":
                            boxes.append(tuple(map(int, box.xyxy[0])))
            except Exception as e:
                print(f"[PERSON MODEL ERROR] {e}")
            with self.cv: self.results[ci] = (boxes, frame, ts, raw)

def process_cam_frame(frame, ci, zone_editor, now_ts, detection, pw=None, ph=None):
    """detection = (boxes, frame they were found in, time, full-res frame)
    from PersonDetector. Crops and alert photos come from that same frame;
    the PPE models get their crop from the camera's FULL resolution, since
    gloves and goggles are only a few pixels big at PROC size (a 2520x1080
    camera shrunk to 960x720 keeps 38% of its width). Returns what to draw."""
    if pw is None: pw = frame.shape[1]
    if ph is None: ph = frame.shape[0]
    zones   = [z for z in zone_editor.get_zones(ci) if zone_on(z, now_ts)]
    tracked = cam_tracked[ci]
    boxes = []; raw = None
    if detection is not None:
        boxes, frame = detection[0], detection[1]
        raw = detection[3] if len(detection) > 3 else None
    seen_ids = set()
    draws    = []   # (x1, y1, x2, y2, colour, label) in this frame's pixels
    supported = supported_items()

    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        tid    = get_track(ci, cx, cy)
        t      = tracked[tid]; t["cx"],t["cy"] = cx,cy; seen_ids.add(tid)

        # A person can stand in several zones at once (e.g. a helmet zone
        # overlapping a smoking zone) - they're held to the union of all of
        # their checks, plus whatever a trigger zone they passed armed.
        in_rest = False; zone_checks = set(); trig_checks = set(); entry_zone = None
        for zone in zones:
            if not zone_editor.box_in_zone(zone, x1, y1, x2, y2, pw=pw, ph=ph): continue
            zt = zone["type"]
            if zt == "rest_area":
                in_rest = True
            elif zt == "ppe_check":
                zone_checks |= set(zone.get("checks", []))
                entry_zone = entry_zone or zone["name"]
            elif zt == "ppe_trigger":
                trig_checks |= set(zone.get("checks", []))
                entry_zone = entry_zone or zone["name"]

        if trig_checks:
            t["triggered"] = True; t["trigger_checks"] = t["trigger_checks"] | trig_checks
        in_ppe = bool(zone_checks)
        active = zone_checks | (t["trigger_checks"] if t["triggered"] else set())
        armed  = bool(active)
        if not in_rest:
            t["rest_entry_start"] = None; t["rest_notified"] = False

        if in_rest:
            t["grace_active"]=False; t["violation_start"]=None
            t["ppe_confirmed"]=False; t["telegram_violation_start"]=None
            t["triggered"]=False; t["trigger_checks"]=set(); t["zone_entry_logged"]=False
            t["behavior_hist"].clear()
            bc = THEME["ok"]; lbl = "REST AREA"

            if t["rest_entry_start"] is None:
                t["rest_entry_start"] = now_ts

            if not t["rest_notified"] and now_ts - t["rest_entry_start"] >= TELEGRAM_PPE_DELAY:
                t["rest_notified"] = True
                zone_name = "Rest Area"
                for zone in zones:
                    if zone["type"] == "rest_area" and zone_editor.box_in_zone(zone, x1, y1, x2, y2, pw=pw, ph=ph):
                        zone_name = zone["name"]; break
                add_log(f"CAM{ci+1} ENTRY [{zone_name}]: worker resting [W{tid}]", THEME["ok"])
                log_file("REST_ENTRY", f"cam{ci+1},{zone_name},W{tid}")
                crop = frame[max(0,y1-60):y2, max(0,x1-20):min(frame.shape[1],x2+20)]
                msg = (f"Worker entered {zone_name} CAM{ci+1} [W{tid}]\n"
                       f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                send_telegram_photo(msg, crop)
                record_web_event(msg, crop, color=THEME["ok"], cam=ci, alert=False)

        elif armed and not (active & supported):
            bc = THEME["warn"]; lbl = "NO MODEL FOR THESE CHECKS"

        elif armed:
            evaluable = active & supported
            res = t.pop("ppe_result", None)          # from the PPE worker
            if res is not None:
                t["ppe_busy"] = False; t["cached"] = res
                for i, good in res.items():
                    if CHECK_ITEMS[i]["kind"] == "behavior":
                        t["behavior_hist"][i].append(not good)
            cached = t["cached"]
            waiting = not evaluable <= set(cached)    # first result not back yet
            # Re-check on the normal interval, or right away if this person
            # just walked into a zone asking for something not checked yet.
            # One smoking sighting not yet confirmed: look again sooner, so
            # a real smoker is confirmed (and alerted) within about a second.
            unconfirmed = any(CHECK_ITEMS[i]["kind"] == "behavior" and 0 < sum(t["behavior_hist"][i]) < BEHAVIOR_MIN_HITS
                              for i in evaluable)
            interval = BEHAVIOR_RECHECK if unconfirmed else PPE_INTERVAL
            if not t.get("ppe_busy") and (now_ts - t["last_ppe"] > interval or waiting):
                if raw is not None and raw.shape[1] > frame.shape[1]:
                    sx, sy = raw.shape[1] / frame.shape[1], raw.shape[0] / frame.shape[0]
                    submit_ppe_check(t, raw, (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)), evaluable)
                else:
                    submit_ppe_check(t, frame, (x1, y1, x2, y2), evaluable)
                t["last_ppe"] = now_ts
            failing = set()
            for i in evaluable:
                if CHECK_ITEMS[i]["kind"] == "behavior":
                    if sum(t["behavior_hist"][i]) >= BEHAVIOR_MIN_HITS: failing.add(i)
                elif not cached.get(i, False):
                    failing.add(i)
            ok = not failing
            # Smoking is never "someone adjusting their helmet": no grace
            # period and no Telegram delay - it alerts as soon as confirmed.
            behavior_now = any(CHECK_ITEMS[i]["kind"] == "behavior" for i in failing)
            grace = 0 if behavior_now else GRACE_PERIOD
            alert_delay = 0 if behavior_now else TELEGRAM_PPE_DELAY
            # Seen once but not yet confirmed: that's "still checking", not
            # "OK" - marking it OK would make the confirming sighting look
            # like an OK->violation change and apply the GRACE_PERIOD meant
            # for someone briefly adjusting a helmet.
            pending = waiting or ok and any(CHECK_ITEMS[i]["kind"] == "behavior" and sum(t["behavior_hist"][i]) > 0
                                 for i in evaluable)

            if not t["zone_entry_logged"] and not pending:
                t["zone_entry_logged"] = True
                zone_name = entry_zone or "Inspection zone"
                if ok:
                    add_log(f"CAM{ci+1} ENTRY [{zone_name}]: OK", THEME["ok"])
                    log_file("ENTRY_OK", f"cam{ci+1},{zone_name}")
                else:
                    missing = violation_short(failing)
                    add_log(f"CAM{ci+1} ENTRY [{zone_name}]: {missing}", THEME["danger"])
                    log_file("ENTRY_VIOLATION", f"cam{ci+1},{zone_name},{missing}")

            if pending:
                bc = THEME["warn"]; lbl = "CHECKING"
            elif ok:
                t["ppe_ok"]=True; t["ppe_confirmed"]=True
                t["missing"]=""; t["violation_start"]=None
                t["grace_active"]=False; t["confirm_time"]=now_ts
                t["telegram_violation_start"]=None; t["triggered"]=False
                t["trigger_checks"]=set(); t["violation_alerted"]=False
                bc = THEME["ok"]; lbl = "OK"
            else:
                tag = "" if in_ppe else " [TRIG]"
                if t["ppe_confirmed"]:
                    if t["violation_start"] is None:
                        t["violation_start"]=now_ts; t["grace_active"]=True
                    elapsed = now_ts - t["violation_start"]
                    if elapsed < grace:
                        bc=THEME["warn"]; lbl=f"CHECKING {grace-elapsed:.0f}s{tag}"
                    else:
                        t["ppe_ok"]=False; t["missing"]=violation_short(failing)
                        t["grace_active"]=False; t["violation_start"]=None
                        t["ppe_confirmed"]=False
                        bc=THEME["danger"]; lbl=f"VIOLATION: {t['missing']}{tag}"
                        add_log(f"CAM{ci+1} VIOLATION: {t['missing']}", THEME["danger"])
                        log_file("PPE_VIOLATION", f"cam{ci+1},{t['missing']}")
                        # One alert per continuous tracked violation "episode" -
                        # violation_alerted latches True the first time this
                        # worker's alert actually goes out and is only cleared
                        # by PPE becoming OK or leaving the zone (see below), or
                        # implicitly by tracking loss creating a fresh track (a
                        # new tid never has this flag set). Without this gate,
                        # a worker who just stays in continuous violation would
                        # re-trigger every TELEGRAM_PPE_DELAY seconds forever.
                        if not t["violation_alerted"]:
                            if t["telegram_violation_start"] is None:
                                t["telegram_violation_start"]=now_ts
                            elif now_ts-t["telegram_violation_start"] >= alert_delay:
                                crop=frame[max(0,y1-60):y2,max(0,x1-20):min(frame.shape[1],x2+20)]
                                title, lines = violation_message(failing)
                                msg=(f"{title} CAM{ci+1} [W{tid}]!\n" + "\n".join(lines) +
                                     f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                send_telegram_photo(msg, crop)
                                record_web_event(msg, crop, cam=ci)
                                t["violation_alerted"]=True
                else:
                    t["ppe_ok"]=False; t["missing"]=violation_short(failing)
                    if not t["violation_alerted"]:
                        if t["telegram_violation_start"] is None:
                            t["telegram_violation_start"]=now_ts
                        elif now_ts-t["telegram_violation_start"] >= alert_delay:
                            crop=frame[max(0,y1-60):y2,max(0,x1-20):min(frame.shape[1],x2+20)]
                            title, lines = violation_message(failing)
                            msg=(f"{title} CAM{ci+1} [W{tid}]!\n" + "\n".join(lines) +
                                 f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            send_telegram_photo(msg, crop)
                            record_web_event(msg, crop, cam=ci)
                            t["violation_alerted"]=True
                    bc=THEME["danger"]; lbl=f"VIOLATION: {t['missing']}{tag}"
        else:
            t["telegram_violation_start"]=None; t["zone_entry_logged"]=False
            t["violation_alerted"]=False; t["behavior_hist"].clear()
            bc=THEME["warn"]; lbl="IN FRAME"

        bh=y2-y1; pad_top=max(60,int(bh*0.25)); dy1=max(0,y1-pad_top)
        draws.append((x1, dy1, x2, y2, bc, lbl))

    for tid in list(tracked.keys()):
        if tid not in seen_ids:
            t = tracked[tid]
            if t.get("zone_entry_logged"):
                status = "OK" if t["ppe_ok"] else f"VIOLATION {t['missing']}"
                add_log(f"CAM{ci+1} EXIT: {status}", THEME["ok"] if t["ppe_ok"] else THEME["danger"])
                log_file("EXIT", f"cam{ci+1},{status}")
            del tracked[tid]

    unchecked = set()
    for z in zones:
        if ZONE_TYPES[z["type"]]["has_checks"]:
            unchecked |= set(z.get("checks", [])) - supported
    return draws, unchecked

def draw_person_overlays(tile, draws, sx, sy, compact):
    """Draws process_cam_frame's boxes on a display tile of any size, with
    the label on a solid chip so it stays readable on any background."""
    fs = max(0.34, min(0.6, 0.69 * sx))
    for (x1, y1, x2, y2, col, lbl) in draws:
        X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
        cv2.rectangle(tile, (X1, Y1), (X2, Y2), col, 1 if compact else 2, cv2.LINE_AA)
        if compact or not lbl: continue
        (tw, th), _ = text_size(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ly2 = Y1 if Y1 - th - 8 >= 0 else Y1 + th + 8
        cv2.rectangle(tile, (X1 - 1, ly2 - th - 8), (X1 + tw + 10, ly2), col, -1)
        dark = (0.114*col[0] + 0.587*col[1] + 0.299*col[2]) > 140
        put_text(tile, lbl, (X1 + 5, ly2 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs,
                 THEME["black"] if dark else THEME["white"], 1)

def shade(img, x1, y1, x2, y2, color, alpha=0.7):
    """Semi-transparent filled box - tile headers/badges read as overlays
    on the video instead of hard blocks."""
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x1 >= x2 or y1 >= y2: return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)

STRIP_GAP      = 8
COMPACT_TILE_H = 170    # tiles shorter than this get the slim header and no labels

def compute_tiles(n, cols, rows, PW, PH, focus, aspect):
    """Where each camera goes in the grid area: rects[ci] = (x, y, w, h).
    Normal view is the plain grid; with focus set (double-click) that camera
    is shown large, keeping its own aspect ratio, and the others sit in a
    strip underneath. Returns (rects, grid_w, grid_h, strip_y or None)."""
    gw, gh = cols*PW, rows*PH
    if focus is None:
        return [((i % cols)*PW, (i // cols)*PH, PW, PH) for i in range(n)], gw, gh, None
    others = [i for i in range(n) if i != focus]
    k = len(others); g = STRIP_GAP
    sh = max(80, min(160, gh // 5)); sw = sh * 4 // 3
    if k*sw + (k+1)*g > gw:
        sw = max(48, (gw - (k+1)*g) // k); sh = sw * 3 // 4
    strip_y = gh - sh - 2*g
    rects = [None]*n
    x0 = max(g, (gw - (k*sw + (k-1)*g)) // 2)
    for j, i in enumerate(others):
        rects[i] = (x0 + j*(sw+g), strip_y + g, sw, sh)
    avail_w, avail_h = gw - 2*g, strip_y - 2*g
    bw = min(avail_w, int(avail_h * aspect)); bh = min(avail_h, int(bw / aspect))
    rects[focus] = ((gw - bw) // 2, g + (avail_h - bh) // 2, bw, bh)
    return rects, gw, gh, strip_y

def add_entry_context_menu(entry):
    """tk.Entry has NO right-click menu by default — only bare Ctrl+V paste,
    which isn't obvious. This adds Cut/Copy/Paste/Select All on right-click,
    the way every other app behaves, so pasting a Telegram token/URL just
    works the way people expect."""
    menu = tk.Menu(entry, tearoff=0, bg=UI["panel"], fg=UI["text"],
                    activebackground=UI["accent"], activeforeground="#0c0f13")
    menu.add_command(label="Cut",        command=lambda: entry.event_generate("<<Cut>>"))
    menu.add_command(label="Copy",       command=lambda: entry.event_generate("<<Copy>>"))
    menu.add_command(label="Paste",      command=lambda: entry.event_generate("<<Paste>>"))
    menu.add_separator()
    menu.add_command(label="Select All", command=lambda: entry.select_range(0, "end"))
    def _show(event):
        try: menu.tk_popup(event.x_root, event.y_root)
        finally: menu.grab_release()
    entry.bind("<Button-3>", _show)
    return entry

def mask_camera_source(src):
    """Hides an embedded rtsp://user:pass@... password so it never shows up
    in the on-screen camera list during a live demo (still stored in full in
    system_config.json, same as the Telegram token already is)."""
    src = str(src)
    if "://" not in src or "@" not in src: return src
    try:
        scheme, rest = src.split("://", 1)
        creds, host = rest.split("@", 1)
        if ":" in creds:
            user, _pw = creds.split(":", 1)
            creds = f"{user}:****"
        return f"{scheme}://{creds}@{host}"
    except Exception:
        return src

class IPCameraDialog(tk.Toplevel):
    """Add a networked camera (NVR channel, IP camera, ONVIF) by RTSP/HTTP
    URL, with a background-threaded 'Test Connection' so a bad IP or an
    offline camera never freezes the dialog while probing it."""
    EXAMPLES = ("Hikvision:  rtsp://user:pass@IP:554/Streaming/Channels/101\n"
                "Dahua:      rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0\n"
                "Generic ONVIF:  rtsp://user:pass@IP:554/onvif1")

    def __init__(self, parent):
        super().__init__(parent)
        self.result=None; self.title("Add IP Camera")
        self.configure(bg=UI["panel"]); self.resizable(False,False)
        self.transient(parent); self.grab_set()
        self._test_result={"done":True,"ok":None}
        self._build()
        self.update_idletasks()
        px=parent.winfo_rootx()+parent.winfo_width()//2-self.winfo_width()//2
        py=parent.winfo_rooty()+parent.winfo_height()//2-self.winfo_height()//2
        self.geometry(f"+{max(px,0)}+{max(py,0)}")
        self.protocol("WM_DELETE_WINDOW",self._cancel); self.wait_window(self)

    def _build(self):
        pad=dict(padx=18,pady=(14,4))
        tk.Label(self,text="Name",bg=UI["panel"],fg=UI["text_dim"],
                 font=("Segoe UI",9)).pack(anchor="w",**pad)
        self.name_var=tk.StringVar(value="NVR CAM")
        add_entry_context_menu(tk.Entry(self,textvariable=self.name_var,bg=UI["field"],
                 fg=UI["text"],insertbackground=UI["text"],relief="flat",bd=6,width=48)
                 ).pack(fill="x",padx=18)
        tk.Label(self,text="Stream URL (RTSP/HTTP)",bg=UI["panel"],fg=UI["text_dim"],
                 font=("Segoe UI",9)).pack(anchor="w",**pad)
        self.url_var=tk.StringVar(value="rtsp://")
        add_entry_context_menu(tk.Entry(self,textvariable=self.url_var,bg=UI["field"],
                 fg=UI["text"],insertbackground=UI["text"],relief="flat",bd=6,width=48)
                 ).pack(fill="x",padx=18)
        tk.Label(self,text=self.EXAMPLES,bg=UI["panel"],fg=UI["text_dim"],
                 font=("Consolas",8),justify="left").pack(anchor="w",padx=18,pady=(6,4))
        trow=tk.Frame(self,bg=UI["panel"]); trow.pack(fill="x",padx=18,pady=(4,4))
        self.test_btn=tk.Button(trow,text="Test Connection",command=self._test_connection,
                    bg=UI["accent"],fg="#0c0f13",activebackground=UI["accent_hi"],
                    activeforeground="#0c0f13",relief="flat",padx=10,pady=5,bd=0,
                    font=("Segoe UI",9,"bold"),cursor="hand2")
        self.test_btn.pack(side="left")
        self.test_lbl=tk.Label(trow,text="Not tested",bg=UI["panel"],fg=UI["text_dim"],
                                font=("Segoe UI",9))
        self.test_lbl.pack(side="left",padx=(10,0))
        tk.Frame(self,bg=UI["border"],height=1).pack(fill="x",padx=18,pady=(10,6))
        br=tk.Frame(self,bg=UI["panel"]); br.pack(fill="x",padx=18,pady=(0,16))
        tk.Button(br,text="Save",command=self._save,bg=UI["success"],fg="#0c0f13",
                  relief="flat",bd=0,padx=14,pady=6,font=("Segoe UI",9,"bold"),
                  cursor="hand2").pack(side="right")
        tk.Button(br,text="Cancel",command=self._cancel,bg=UI["field"],fg=UI["text"],
                  relief="flat",bd=0,padx=14,pady=6,font=("Segoe UI",9),
                  cursor="hand2").pack(side="right",padx=(0,8))

    def _test_connection(self):
        url=self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Empty URL","Enter a stream URL first."); return
        self.test_lbl.config(text="Testing...",fg=UI["text_dim"])
        self.test_btn.config(state="disabled")
        self._test_result={"done":False,"ok":None}
        def worker():
            ok=False
            try:
                cap=open_capture(url,timeout=6.0)
                if cap is not None:
                    ret,_=cap.read()
                    ok=bool(ret)
                    cap.release()
            except Exception:
                ok=False
            self._test_result={"done":True,"ok":ok}
        threading.Thread(target=worker,daemon=True).start()
        self.after(200,self._poll_test)

    def _poll_test(self):
        if not self._test_result["done"]:
            self.after(200,self._poll_test); return
        self.test_btn.config(state="normal")
        if self._test_result["ok"]:
            self.test_lbl.config(text="Connected",fg=UI["success"])
        else:
            self.test_lbl.config(text="Failed - check URL / network / credentials",fg=UI["danger"])

    def _save(self):
        name=self.name_var.get().strip(); url=self.url_var.get().strip()
        if not url or url=="rtsp://":
            messagebox.showerror("Missing URL","Please enter a stream URL."); return
        if "://" not in url:
            if not messagebox.askyesno("Unusual URL",
                    "This doesn't look like a stream URL (expected rtsp://... or "
                    "http://...). Save it anyway?"): return
        self.result={"source":url,"name":name or "IP CAM"}
        self.destroy()

    def _cancel(self):
        self.result=None
        self.destroy()

# ── SETUP GUI ─────────────────────────────────────────────────────────────────
UI = {
    "bg":"#121214","panel":"#1a1a1e","field":"#232328","border":"#37373a",
    "accent":"#2196f3","accent_hi":"#1e88e5","danger":"#eb4646","danger_hi":"#d63c3c",
    "success":"#5ac878","success_hi":"#4bb468","text":"#e6e6eb","text_dim":"#96969b",
}
# Tk can use the same Windows icon font as the monitor's top bar.
TK_ICON_CODES = {"camera": 0xE714, "telegram": 0xE724, "clock": 0xE823,
                 "shield": 0xEA18, "globe": 0xE774}

class SetupApp:
    def __init__(self):
        self.cfg=load_config(); self.launched=False
        self.root=tk.Tk()
        self.root.title("Safety Cam — Setup")
        self.root.configure(bg=UI["bg"])
        if os.path.exists(ICON_PATH):
            try: self.root.iconbitmap(default=ICON_PATH)
            except: pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        # Size the window to fit whatever sections _build() packed instead of
        # a hardcoded height — the REMOTE ACCESS section added enough content
        # that a fixed 740x620 was clipping the Launch button off the bottom.
        self.root.update_idletasks()
        w = max(740, self.root.winfo_reqwidth())
        h = self.root.winfo_reqheight()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{min(h, sh-80)}")

    def _on_close(self):
        self.launched = False
        self.root.destroy()

    def _btn(self,p,t,cmd,kind="accent"):
        bg={"accent":UI["accent"],"danger":UI["danger"],"success":UI["success"]}[kind]
        hi={"accent":UI["accent_hi"],"danger":UI["danger_hi"],"success":UI["success_hi"]}[kind]
        fg="#0c0f13" if kind=="success" else "#ffffff"
        b=tk.Button(p,text=t,command=cmd,bg=bg,fg=fg,activebackground=hi,
                    activeforeground=fg,relief="flat",padx=10,pady=6,bd=0,
                    font=("Segoe UI",9,"bold"),cursor="hand2")
        b.bind("<Enter>",lambda e:b.config(bg=hi)); b.bind("<Leave>",lambda e:b.config(bg=bg))
        return b

    def _icon(self,parent,kind,size=18,color=None,bg=None):
        """Small hand-drawn glyph on a tk.Canvas — same vector-icon approach
        as the OpenCV monitor window, so no external image assets are needed
        and nothing has to be added to the PyInstaller .spec datas=[]."""
        color=color or UI["accent"]; bg=bg or UI["panel"]
        try:
            import tkinter.font as tkfont
            fams = set(tkfont.families(parent))
            fam = next((f for f in ("Segoe Fluent Icons", "Segoe MDL2 Assets") if f in fams), None)
        except Exception:
            fam = None
        if fam and kind in TK_ICON_CODES:
            return tk.Label(parent, text=chr(TK_ICON_CODES[kind]), font=(fam, max(8, int(size*0.6))),
                            fg=color, bg=bg)
        c=tk.Canvas(parent,width=size,height=size,bg=bg,highlightthickness=0)
        m=size/2
        if kind=="camera":
            c.create_rectangle(2,m-4,size-2,size-2,outline=color,width=2)
            c.create_rectangle(m-5,1,m+4,m-4,outline=color,width=2)
            c.create_oval(m-4,m-2,m+4,m+6,outline=color,width=2)
        elif kind=="telegram":
            c.create_polygon(2,m, size-2,2, m+3,size-2, m-2,m+3, 2,m,
                              outline=color,fill="",width=2)
        elif kind=="clock":
            c.create_oval(2,2,size-2,size-2,outline=color,width=2)
            c.create_line(m,m,m,5,fill=color,width=2)
            c.create_line(m,m,size-5,m+2,fill=color,width=2)
        elif kind=="shield":
            c.create_polygon(m,2, size-2,6, size-3,m+2, m,size-2, 3,m+2, 2,6,
                              outline=color,fill="",width=2)
            c.create_line(m-4,m,m-1,m+4,fill=color,width=2)
            c.create_line(m-1,m+4,m+5,m-4,fill=color,width=2)
        elif kind=="globe":
            c.create_oval(2,2,size-2,size-2,outline=color,width=2)
            c.create_line(2,m,size-2,m,fill=color,width=1)
            c.create_oval(m-3,2,m+3,size-2,outline=color,width=1)
        return c

    def _section(self,title,icon=None):
        f=tk.Frame(self.root,bg=UI["panel"],highlightbackground=UI["border"],highlightthickness=1)
        f.pack(fill="x",padx=16,pady=6)
        tk.Frame(f,bg=UI["accent"],height=2).pack(fill="x",side="top")
        hrow=tk.Frame(f,bg=UI["panel"]); hrow.pack(anchor="w",padx=12,pady=(10,4))
        if icon: self._icon(hrow,icon,16,UI["accent"]).pack(side="left",padx=(0,6))
        tk.Label(hrow,text=title,bg=UI["panel"],fg=UI["accent"],
                 font=("Segoe UI",10,"bold")).pack(side="left")
        return f

    def _build(self):
        hdr=tk.Frame(self.root,bg=UI["bg"]); hdr.pack(fill="x",pady=(18,8))
        trow=tk.Frame(hdr,bg=UI["bg"]); trow.pack()
        self._icon(trow,"shield",22,UI["accent"],bg=UI["bg"]).pack(side="left",padx=(0,8))
        tk.Label(trow,text="SAFETY CAM",bg=UI["bg"],fg=UI["accent"],
                 font=("Segoe UI",16,"bold")).pack(side="left")
        tk.Label(hdr,text="Surveillance setup & zone configuration",
                 bg=UI["bg"],fg=UI["text_dim"],font=("Segoe UI",9)).pack(pady=(2,0))
        cf=self._section("CAMERAS",icon="camera")
        row=tk.Frame(cf,bg=UI["panel"]); row.pack(fill="x",padx=12,pady=(0,12))
        self.cam_lb=tk.Listbox(row,bg=UI["field"],fg=UI["text"],height=4,bd=0,
                               highlightthickness=0,selectbackground=UI["accent"],
                               selectforeground="#ffffff",font=("Consolas",9))
        self.cam_lb.pack(side="left",fill="both",expand=True,padx=(0,8))
        self._refresh_cams()
        bf=tk.Frame(row,bg=UI["panel"]); bf.pack(side="left")
        self._btn(bf,"+ Live Camera",self._add_live).pack(fill="x",pady=2)
        self._btn(bf,"+ Video File", self._add_video).pack(fill="x",pady=2)
        self._btn(bf,"+ IP Camera",  self._add_ip).pack(fill="x",pady=2)
        self._btn(bf,"Remove",       self._rem_cam,"danger").pack(fill="x",pady=2)
        tf=self._section("TELEGRAM ALERTS",icon="telegram")
        body=tk.Frame(tf,bg=UI["panel"]); body.pack(fill="x",padx=12,pady=(0,12))
        for r,(lbl,var_name,w) in enumerate([("Bot Token","tok",48),("Chat ID","cid",30)]):
            tk.Label(body,text=lbl,bg=UI["panel"],fg=UI["text_dim"],
                     font=("Segoe UI",10)).grid(row=r,column=0,sticky="w",padx=(0,8),pady=4)
            setattr(self,var_name,tk.StringVar(value=self.cfg.get(
                "telegram_token" if var_name=="tok" else "telegram_chat_id","")))
            add_entry_context_menu(tk.Entry(body,textvariable=getattr(self,var_name),width=w,
                     bg=UI["field"],fg=UI["text"],insertbackground=UI["text"],relief="flat",
                     bd=6)).grid(row=r,column=1,pady=4,sticky="w")
        rf=self._section("REMOTE ACCESS",icon="globe")
        rbody=tk.Frame(rf,bg=UI["panel"]); rbody.pack(fill="x",padx=12,pady=(0,4))
        self.rem_user=tk.StringVar(value=self.cfg.get("remote_user",""))
        self.rem_pass=tk.StringVar(value="")
        for r,(lbl,var,show) in enumerate([("Username",self.rem_user,None),
                                            ("Password",self.rem_pass,"*")]):
            tk.Label(rbody,text=lbl,bg=UI["panel"],fg=UI["text_dim"],
                     font=("Segoe UI",10)).grid(row=r,column=0,sticky="w",padx=(0,8),pady=4)
            add_entry_context_menu(tk.Entry(rbody,textvariable=var,width=30,bg=UI["field"],
                     fg=UI["text"],insertbackground=UI["text"],relief="flat",bd=6,
                     show=show or "")).grid(row=r,column=1,pady=4,sticky="w")
        self.pass_status_lbl=tk.Label(rbody,text="", bg=UI["panel"],font=("Segoe UI",8))
        self.pass_status_lbl.grid(row=1,column=2,sticky="w",padx=(8,0))
        self._refresh_pass_status()
        self.remote_enabled_var=tk.BooleanVar(value=self.cfg.get("remote_enabled",False))
        tk.Checkbutton(rf,text="Enable remote access (local dashboard + Cloudflare Tunnel)",
                       variable=self.remote_enabled_var,command=self._on_remote_toggle,
                       bg=UI["panel"],fg=UI["text"],
                       selectcolor=UI["field"],activebackground=UI["panel"],
                       activeforeground=UI["text"],font=("Segoe UI",9)
                       ).pack(anchor="w",padx=12,pady=(0,2))
        tk.Label(rf,text="Password field stays blank for privacy — leave it empty to keep "
                          "the previously saved password.",
                 bg=UI["panel"],fg=UI["text_dim"],font=("Segoe UI",8)
                 ).pack(anchor="w",padx=12,pady=(0,6))
        lrow=tk.Frame(rf,bg=UI["panel"]); lrow.pack(fill="x",padx=12,pady=(0,4))
        tk.Label(lrow,text="Public link",bg=UI["panel"],fg=UI["text_dim"],
                 font=("Segoe UI",10)).pack(side="left",padx=(0,8))
        self.remote_link_var=tk.StringVar(value="not started")
        self.remote_link_entry=tk.Entry(lrow,textvariable=self.remote_link_var,width=38,
                 bg=UI["field"],fg=UI["text"],insertbackground=UI["text"],relief="flat",
                 bd=6,state="readonly",readonlybackground=UI["field"])
        add_entry_context_menu(self.remote_link_entry).pack(side="left",fill="x",expand=True)
        self._btn(lrow,"Copy",self._copy_remote_link).pack(side="left",padx=(6,0))
        tk.Label(rf,text="This link changes every time the program starts (no fixed domain "
                          "configured) — copy it fresh each session.",
                 bg=UI["panel"],fg=UI["text_dim"],font=("Segoe UI",8)
                 ).pack(anchor="w",padx=12,pady=(0,10))
        self._remote_link_copied=False
        if self.remote_enabled_var.get() and self.cfg.get("remote_pass_hash"):
            self._maybe_start_remote_early()
        self._btn(self.root,"▶   LAUNCH MONITOR",self._launch,"success").pack(pady=18)

    def _refresh_cams(self):
        self.cam_lb.delete(0,"end")
        for i,c in enumerate(self.cfg["cameras"]):
            self.cam_lb.insert("end",f"  CAM{i+1}  ·  {c['name']}  ·  [{mask_camera_source(c['source'])}]")

    def _add_live(self):
        if len(self.cfg["cameras"])>=12: messagebox.showwarning("Limit","Max 12 cameras"); return
        idx=simpledialog.askinteger("Index","Camera index:",minvalue=0,maxvalue=9)
        if idx is None: return
        name=simpledialog.askstring("Name","Name:",initialvalue=f"CAM{len(self.cfg['cameras'])+1}")
        self.cfg["cameras"].append({"source":str(idx),"name":name or f"CAM{len(self.cfg['cameras'])+1}"})
        save_config(self.cfg); self._refresh_cams()

    def _add_video(self):
        if len(self.cfg["cameras"])>=12: messagebox.showwarning("Limit","Max 12 cameras"); return
        path=filedialog.askopenfilename(filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not path: return
        name=simpledialog.askstring("Name","Name:",initialvalue=os.path.basename(path).split(".")[0])
        self.cfg["cameras"].append({"source":path,"name":name or "CAM"})
        save_config(self.cfg); self._refresh_cams()

    def _add_ip(self):
        if len(self.cfg["cameras"])>=12: messagebox.showwarning("Limit","Max 12 cameras"); return
        dlg=IPCameraDialog(self.root)
        if dlg.result is None: return
        self.cfg["cameras"].append(dlg.result)
        save_config(self.cfg); self._refresh_cams()

    def _rem_cam(self):
        sel=self.cam_lb.curselection()
        if not sel: return
        self.cfg["cameras"].pop(sel[0]); save_config(self.cfg); self._refresh_cams()

    def _refresh_pass_status(self):
        if self.cfg.get("remote_pass_hash"):
            self.pass_status_lbl.config(text="✓ saved", fg=UI["success"])
        else:
            self.pass_status_lbl.config(text="not set", fg=UI["text_dim"])

    def _on_remote_toggle(self):
        if self.remote_enabled_var.get():
            self._maybe_start_remote_early()

    def _maybe_start_remote_early(self):
        """Starts the tunnel as soon as it's turned on in Setup, instead of
        waiting for Launch — so the public link (and the fact that it's a
        fresh one this run) is visible and copyable right here, in a real
        Tk field, instead of only as unselectable pixels in the OpenCV log."""
        user = self.rem_user.get().strip()
        new_pass = self.rem_pass.get()
        if new_pass and generate_password_hash:
            self.cfg["remote_pass_hash"] = generate_password_hash(new_pass)
            self._refresh_pass_status()
        if not user or not self.cfg.get("remote_pass_hash"):
            self.remote_link_var.set("set a username + password first")
            return
        self.cfg["remote_user"] = user
        self.cfg["remote_enabled"] = True
        save_config(self.cfg)
        self.remote_link_var.set("starting...")
        start_remote_access(self.cfg)
        self.root.after(1000, self._poll_remote_link)

    def _poll_remote_link(self):
        url = _remote_state.get("url")
        if url:
            self.remote_link_var.set(url)
            if not self._remote_link_copied:
                self._remote_link_copied = True
                try:
                    self.root.clipboard_clear(); self.root.clipboard_append(url)
                except Exception: pass
                add_log(f"Public link ready (copied to clipboard): {url}", THEME["accent"])
            return
        if _remote_state.get("flask_started"):
            self.remote_link_var.set("connecting tunnel...")
        self.root.after(1000, self._poll_remote_link)

    def _copy_remote_link(self):
        url = _remote_state.get("url")
        if not url:
            messagebox.showinfo("Remote Access",
                "No public link yet — enable remote access and wait a few seconds "
                "for the tunnel to connect.")
            return
        self.root.clipboard_clear(); self.root.clipboard_append(url)

    def _launch(self):
        self.cfg["telegram_token"]  =self.tok.get().strip()
        self.cfg["telegram_chat_id"]=self.cid.get().strip()
        self.cfg["remote_user"]     =self.rem_user.get().strip()
        self.cfg["remote_enabled"]  =self.remote_enabled_var.get()
        new_pass = self.rem_pass.get()
        if new_pass:
            if generate_password_hash:
                self.cfg["remote_pass_hash"] = generate_password_hash(new_pass)
            else:
                messagebox.showwarning("Remote Access",
                    "Flask/Werkzeug isn't installed - remote access will stay disabled.")
        if self.remote_enabled_var.get() and not self.cfg.get("remote_pass_hash"):
            messagebox.showwarning("Remote Access",
                "Remote access is enabled but no password has ever been set — "
                "it will stay off until you set one.")

        save_config(self.cfg); self.launched=True; self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.cfg if self.launched else None

# ── REMOTE ACCESS (local web dashboard + Cloudflare Tunnel) ────────────────────
# Optional layer: serves the exact same combined grid+bar image the OpenCV
# window shows, as an MJPEG stream behind a login, over a normal HTTP server.
# Whichever machine runs this app just opens an outbound tunnel to reach the
# same kind of public link — no port-forwarding, no fixed IP needed, and the
# tunnel provides HTTPS to the browser even though Flask itself only ever
# speaks plain HTTP on localhost.
REMOTE_PORT = 8787
_frame_lock = threading.Lock()
latest_frame_jpeg = None
_remote_state = {"flask_started": False, "cloudflared_proc": None, "url": None, "last_encode": 0.0,
                  "watchdog_started": False}
WEB_ENCODE_INTERVAL = 0.15  # ~6-7fps for the outbound web stream, independent of the local display fps

# cloudflared's own local diagnostics server (always on 127.0.0.1:20241 by
# default) - used to detect when the tunnel process is alive but has quietly
# lost its connection to Cloudflare's edge (see _tunnel_watchdog).
TUNNEL_METRICS_PORT   = 20241
TUNNEL_CHECK_INTERVAL = 15   # seconds between health checks
TUNNEL_UNREADY_LIMIT  = 3    # consecutive failed checks before restarting the tunnel

# Each viewer's MJPEG generator times how long its own socket write actually
# takes (see _mjpeg_generator) and folds that into this running average —
# a slow/congested connection makes writes take longer, so this is a decent
# proxy for "how much bandwidth is actually available right now" without
# needing a real ABR protocol. publish_web_frame reads it to pick quality.
_net_stats = {"avg_write_ms": 25.0}
_net_stats_lock = threading.Lock()

def _pick_quality_and_scale():
    # Thresholds are calibrated for real internet round-trips through the
    # tunnel (typically 100-400ms even on a perfectly fine connection), not
    # localhost — treating that as "slow" was making the stream shrink for
    # everyone. Only genuinely struggling connections drop resolution now.
    with _net_stats_lock:
        ms = _net_stats["avg_write_ms"]
    if   ms < 250:  return 80, 0.6
    elif ms < 600:  return 65, 0.5
    elif ms < 1200: return 50, 0.4
    elif ms < 2500: return 35, 0.3
    else:          return 22, 0.2    # very slow: keep it moving over clarity

# ── WEB EVENTS (violations / rest-area entries, shown on the dashboard
# itself instead of only Telegram) ──────────────────────────────────────────
WEB_EVENTS_MAX = 30
web_events = []
_web_events_lock = threading.Lock()
_web_event_seq = [0]

# ── DAILY ALERTS ──────────────────────────────────────────────────────────────
# Every alert photo is also saved to alerts/<day>/ with a line in
# alerts.jsonl (time, camera, title, file) - the Admin Log's ALERTS tab
# shows them per day, and a double-click plays that camera's recording
# from just before the alert.
ALERTS_DIR = "alerts"
_alert_days = {}
_alert_lock = threading.Lock()

def save_alert(cam, ts, msg, image):
    try:
        d = datetime.fromtimestamp(ts); day = d.strftime("%Y-%m-%d")
        folder = os.path.join(ALERTS_DIR, day); os.makedirs(folder, exist_ok=True)
        fname = f"{d.strftime('%H%M%S_%f')}_cam{cam+1}.jpg"
        ok, enc = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok: return
        with open(os.path.join(folder, fname), "wb") as f: f.write(enc.tobytes())
        entry = {"ts": ts, "cam": cam, "title": msg.split("\n")[0], "file": fname}
        with open(os.path.join(folder, "alerts.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        with _alert_lock:
            if day in _alert_days: _alert_days[day].append(entry)
    except Exception as e:
        print(f"[ALERT SAVE ERROR] {e}")

def load_alerts(day):
    with _alert_lock:
        if day in _alert_days: return _alert_days[day]
    entries = []
    try:
        with open(os.path.join(ALERTS_DIR, day, "alerts.jsonl"), encoding="utf-8") as f:
            for line in f:
                try: entries.append(json.loads(line))
                except ValueError: pass
    except OSError:
        pass
    with _alert_lock:
        _alert_days.setdefault(day, entries)
        return _alert_days[day]

def alert_days():
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        days = {d for d in os.listdir(ALERTS_DIR) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)}
    except OSError:
        days = set()
    days.add(today)
    return sorted(days)

def record_web_event(msg, image, color=None, cam=None, alert=True):
    """Mirrors what gets sent to Telegram: the photo goes to the web
    dashboard's in-memory gallery (polled via /events), the text to the
    Admin Log, and - for alerts (alert=True with a camera) - to the daily
    alerts folder shown in the Admin Log's ALERTS tab."""
    if image is None or image.size == 0: return
    try:
        ok, enc = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok: return
        ts = datetime.now().strftime("%H:%M:%S")
        with _web_events_lock:
            _web_event_seq[0] += 1
            eid = _web_event_seq[0]
            web_events.append({"id": eid, "ts": ts, "msg": msg, "jpeg": enc.tobytes()})
            if len(web_events) > WEB_EVENTS_MAX:
                del web_events[:len(web_events)-WEB_EVENTS_MAX]
        admin_log.append({"text": f"[{ts}] " + msg.replace("\n", " | "), "color": color or THEME["danger"]})
        if cam is not None and alert:
            save_alert(cam, time.time(), msg, image)
        if len(admin_log) > ADMIN_LOG_MAX:
            del admin_log[:len(admin_log)-ADMIN_LOG_MAX]
        log_scroll_pos[0] = 0
    except Exception as e:
        print(f"[WEB EVENT ERROR] {e}")

if FLASK_AVAILABLE:
    web_app = Flask(__name__)
    web_app.secret_key = os.urandom(24)
    web_app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

    LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Safety Cam - Login</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{background:#11151c;color:#e8eaed;font-family:'Segoe UI',Arial,sans-serif;
     display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{background:#171c26;border:1px solid #2a323d;border-radius:10px;padding:32px;width:280px}
h1{color:#3fd6c8;font-size:16px;margin:0 0 20px;letter-spacing:.5px}
input{width:100%;box-sizing:border-box;background:#1f2630;border:none;color:#e8eaed;
      padding:10px;border-radius:6px;margin-bottom:12px;font-size:14px}
button{width:100%;background:#3fd6c8;color:#0c0f13;border:none;padding:10px;
       border-radius:6px;font-weight:bold;cursor:pointer;font-size:14px}
.err{color:#e85d6b;font-size:13px;margin-bottom:12px}
</style></head><body>
<form class="card" method="post">
<h1>SAFETY CAM</h1>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<input name="username" placeholder="Username" autofocus autocomplete="username">
<input name="password" type="password" placeholder="Password" autocomplete="current-password">
<button type="submit">Sign in</button>
</form></body></html>"""

    STREAM_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Safety Cam</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden;background:#11151c;color:#e8eaed;font-family:'Segoe UI',Arial,sans-serif}
body{display:flex;flex-direction:column}
header{flex:0 0 auto;padding:8px 14px;background:#171c26;border-bottom:1px solid #2a323d;
       display:flex;justify-content:space-between;align-items:center}
h1{font-size:13px;color:#3fd6c8;margin:0;letter-spacing:.5px}
a{color:#8a9099;font-size:12px;text-decoration:none}
main{flex:1 1 auto;min-height:0;display:flex;flex-direction:row}
.video-wrap{flex:1 1 auto;min-width:0;display:flex;align-items:center;justify-content:center;background:#000}
.video-wrap img{width:100%;height:100%;object-fit:contain;display:block}
.events{flex:0 0 300px;width:300px;background:#171c26;border-left:1px solid #2a323d;
        padding:10px;display:flex;flex-direction:column;gap:10px;overflow-y:auto;overflow-x:hidden}
.events-title{font-size:11px;color:#8a9099;letter-spacing:.5px;margin:0 0 2px}
.events.empty .events-title{display:none}
.events-empty-msg{color:#6b7280;font-size:12px;text-align:center;margin-top:24px}
.events:not(.empty) .events-empty-msg{display:none}
.ev-card{flex:0 0 auto;background:#1f2630;border-radius:8px;overflow:hidden;border:1px solid #2a323d}
.ev-card img{width:100%;height:170px;object-fit:contain;display:block;background:#000}
.ev-card .cap{font-size:12px;padding:8px 10px 2px;color:#e8eaed;white-space:pre-wrap;word-break:break-word}
.ev-card .ts{font-size:10px;color:#6b7280;padding:0 10px 8px}
@media (max-width:760px){
  main{flex-direction:column}
  .events{flex:0 0 auto;width:100%;flex-direction:row;overflow-x:auto;overflow-y:hidden;border-left:none;border-top:1px solid #2a323d}
  .ev-card{flex:0 0 170px;width:170px}
  .ev-card img{height:110px}
}
.expired-overlay{position:fixed;inset:0;background:rgba(10,12,16,.94);display:none;
                  align-items:center;justify-content:center;z-index:50;text-align:center;padding:20px}
.expired-overlay.show{display:flex}
.expired-title{font-size:22px;font-weight:600;color:#e85d6b;margin-bottom:10px}
.expired-sub{font-size:13px;color:#8a9099}
</style></head><body>
<header><h1>SAFETY CAM — LIVE</h1><a href="/logout">Sign out</a></header>
<main>
<div class="video-wrap"><img id="streamImg" src="/stream.mjpg"></div>
<div class="events empty" id="events">
  <div class="events-title">ALERTS</div>
  <div class="events-empty-msg">No alerts yet</div>
</div>
</main>
<div class="expired-overlay" id="expiredOverlay">
  <div>
    <div class="expired-title">تم انتهاء صلاحية هذا الرابط</div>
    <div class="expired-sub">This link has expired — ask the operator for the new one.</div>
  </div>
</div>
<script>
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
// The /events poll below is also this page's heartbeat: once the tunnel
// behind THIS specific link is torn down (a new one was issued elsewhere),
// every request from an already-open tab starts failing at the network
// level - a few consecutive failures means the link is dead, not a blip.
let failCount = 0;
const FAIL_LIMIT = 3;
let expired = false;
async function poll(){
  if (expired) return;
  try{
    const r = await fetch('/events');
    if(!r.ok) throw new Error('bad status '+r.status);
    const d = await r.json();
    failCount = 0;
    const el = document.getElementById('events');
    if(!d.events || !d.events.length){ el.classList.add('empty'); return; }
    el.classList.remove('empty');
    el.innerHTML = '<div class="events-title">ALERTS</div>' + d.events.map(e =>
      '<div class="ev-card"><img src="/event_image/'+e.id+'" loading="lazy">'+
      '<div class="cap">'+esc(e.msg)+'</div><div class="ts">'+esc(e.ts)+'</div></div>'
    ).join('');
  }catch(e){
    failCount++;
    if(failCount >= FAIL_LIMIT){
      expired = true;
      document.getElementById('expiredOverlay').classList.add('show');
      document.getElementById('streamImg').src = '';
    }
  }
}
poll(); setInterval(poll, 4000);
</script>
</body></html>"""

    def _remote_login_ok(username, password):
        cfg = load_config()
        exp_user = cfg.get("remote_user", "")
        exp_hash = cfg.get("remote_pass_hash", "")
        if not exp_user or not exp_hash: return False
        return username == exp_user and check_password_hash(exp_hash, password)

    @web_app.route("/login", methods=["GET", "POST"])
    def _route_login():
        error = None
        if request.method == "POST":
            if _remote_login_ok(request.form.get("username", ""), request.form.get("password", "")):
                session["auth"] = True
                return redirect(url_for("_route_index"))
            time.sleep(1)  # light brute-force slowdown, not real rate limiting
            error = "Invalid username or password"
        return render_template_string(LOGIN_HTML, error=error)

    @web_app.route("/logout")
    def _route_logout():
        session.clear()
        return redirect(url_for("_route_login"))

    @web_app.route("/")
    def _route_index():
        if not session.get("auth"): return redirect(url_for("_route_login"))
        return render_template_string(STREAM_HTML)

    def _mjpeg_generator():
        while True:
            with _frame_lock:
                frame_bytes = latest_frame_jpeg
            if frame_bytes:
                t0 = time.time()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                # the generator only regains control once Werkzeug has
                # actually written the chunk to the socket, so this is a
                # real (if approximate) read on this viewer's connection speed
                write_ms = (time.time()-t0)*1000
                if write_ms < 5000:
                    with _net_stats_lock:
                        _net_stats["avg_write_ms"] = _net_stats["avg_write_ms"]*0.85 + write_ms*0.15
            time.sleep(0.1)  # caps the web stream around ~10fps independent of local display

    @web_app.route("/stream.mjpg")
    def _route_stream():
        if not session.get("auth"): return redirect(url_for("_route_login"))
        return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @web_app.route("/events")
    def _route_events():
        if not session.get("auth"): return redirect(url_for("_route_login"))
        with _web_events_lock:
            items = [{"id": e["id"], "ts": e["ts"], "msg": e["msg"]} for e in reversed(web_events)]
        return {"events": items}

    @web_app.route("/event_image/<int:eid>")
    def _route_event_image(eid):
        if not session.get("auth"): return redirect(url_for("_route_login"))
        with _web_events_lock:
            match = next((e for e in web_events if e["id"] == eid), None)
        if not match: return "", 404
        return Response(match["jpeg"], mimetype="image/jpeg")

def find_cloudflared():
    for c in (os.path.join(BASE_DIR, "cloudflared.exe"), "cloudflared.exe", "cloudflared"):
        if os.path.exists(c): return os.path.abspath(c)
    return shutil.which("cloudflared")

def publish_web_frame(camera_grid):
    """Called once per render loop iteration with the camera-only grid (no
    top bar, no sidebar — those are local operator controls, not something
    a remote viewer needs). Quality/resolution adapt automatically to how
    fast viewers' connections are actually keeping up (see _pick_quality_and_scale)."""
    if not FLASK_AVAILABLE: return
    now = time.time()
    if now - _remote_state["last_encode"] < WEB_ENCODE_INTERVAL: return
    _remote_state["last_encode"] = now
    global latest_frame_jpeg
    try:
        quality, scale = _pick_quality_and_scale()
        # the browser always stretches this to fill the viewer's screen
        # (see .video-wrap img{width:100%;height:100%}), so even the worst
        # tier needs enough real detail not to look absurd blown up on a
        # phone or a big monitor — floor is higher than the old 160x120.
        w = max(480, int(camera_grid.shape[1]*scale))
        h = max(360, int(camera_grid.shape[0]*scale))
        small = cv2.resize(camera_grid, (w, h))
        ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with _frame_lock:
                latest_frame_jpeg = enc.tobytes()
    except Exception as e:
        print(f"[WEB ENCODE ERROR] {e}")

def start_remote_access(cfg):
    if not FLASK_AVAILABLE:
        add_log("Remote access needs Flask (pip install flask) - skipped", THEME["warn"]); return
    if not cfg.get("remote_user") or not cfg.get("remote_pass_hash"):
        add_log("Remote access enabled but no username/password set - skipped", THEME["warn"]); return

    if not _remote_state["flask_started"]:
        def run_flask():
            try:
                web_app.run(host="0.0.0.0", port=REMOTE_PORT, threaded=True,
                            debug=False, use_reloader=False)
            except Exception as e:
                print(f"[REMOTE SERVER ERROR] {e}")
        threading.Thread(target=run_flask, daemon=True).start()
        _remote_state["flask_started"] = True
        add_log(f"Local dashboard: http://localhost:{REMOTE_PORT}", THEME["accent"])

    proc = _remote_state["cloudflared_proc"]
    if proc is not None and proc.poll() is None:
        if _remote_state["url"]:
            add_log(f"Remote dashboard: {_remote_state['url']}", THEME["accent"])
        return

    cloudflared_path = find_cloudflared()
    if not cloudflared_path:
        add_log("cloudflared.exe not found - only the local/LAN dashboard is available. "
                "Get it from github.com/cloudflare/cloudflared/releases for a public link.",
                THEME["warn"])
        return

    def run_tunnel():
        try:
            proc = subprocess.Popen(
                # --protocol http2 forces the tunnel over plain TCP instead of
                # cloudflared's default QUIC/UDP - many routers/campus networks
                # drop long-lived UDP after a while, which silently kills a
                # QUIC tunnel while the process keeps running (looks "alive"
                # but has zero real connections - see _tunnel_watchdog).
                [cloudflared_path, "tunnel", "--protocol", "http2",
                 "--url", f"http://127.0.0.1:{REMOTE_PORT}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                # Without this, launching a console program (cloudflared.exe)
                # from a windowed/no-console app (this one, once built with
                # console=False) makes Windows pop up a brand new visible
                # console window for the child process - that's the "cmd
                # window" that appeared on Launch. This suppresses it; stdout
                # is still captured normally via the pipe above.
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            _remote_state["cloudflared_proc"] = proc
            # Keep draining stdout for the process's whole life (not just
            # until the URL line shows up) so its output pipe never fills up
            # and blocks its own logging/operation.
            for line in proc.stdout:
                if not _remote_state.get("url"):
                    m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
                    if m:
                        _remote_state["url"] = m.group(0)
                        add_log(f"Remote dashboard: {m.group(0)}", THEME["accent"])
        except Exception as e:
            add_log(f"Could not start remote tunnel: {e}", THEME["warn"])
    threading.Thread(target=run_tunnel, daemon=True).start()

    if not _remote_state["watchdog_started"]:
        _remote_state["watchdog_started"] = True
        threading.Thread(target=_tunnel_watchdog, args=(cfg,), daemon=True).start()

def _tunnel_is_ready():
    try:
        r = requests.get(f"http://127.0.0.1:{TUNNEL_METRICS_PORT}/ready", timeout=3)
        return r.json().get("readyConnections", 0) > 0
    except Exception:
        return False

def _tunnel_watchdog(cfg):
    """cloudflared can stay alive as a process while its actual connection to
    Cloudflare's edge is dead (common on networks that drop long-lived UDP) -
    the public link then looks fine from here but returns an error page for
    every real visitor. Poll cloudflared's own /ready endpoint periodically
    and restart the tunnel if it's been disconnected too long, so a demo
    audience never hits a dead link without the presenter noticing."""
    bad_checks = 0
    while True:
        time.sleep(TUNNEL_CHECK_INTERVAL)
        proc = _remote_state.get("cloudflared_proc")
        if proc is None or not _remote_state.get("url"):
            continue
        if proc.poll() is not None:
            bad_checks = 0
            _remote_state["cloudflared_proc"] = None
            _remote_state["url"] = None
            add_log("Remote tunnel process exited - restarting...", THEME["warn"])
            start_remote_access(cfg)
            continue
        if _tunnel_is_ready():
            bad_checks = 0
        else:
            bad_checks += 1
            if bad_checks >= TUNNEL_UNREADY_LIMIT:
                bad_checks = 0
                add_log("Remote tunnel lost connection to Cloudflare - restarting...", THEME["warn"])
                try:
                    proc.terminate()
                except Exception:
                    pass
                _remote_state["cloudflared_proc"] = None
                _remote_state["url"] = None
                start_remote_access(cfg)

@atexit.register
def _cleanup_remote_tunnel():
    proc = _remote_state.get("cloudflared_proc")
    if proc is not None and proc.poll() is None:
        try: proc.terminate()
        except Exception: pass

# ── MAIN MONITOR ──────────────────────────────────────────────────────────────
CAM_OPEN_TIMEOUT     = 6.0   # seconds to wait before giving up on a camera/stream
CAM_RECONNECT_AFTER  = 20    # consecutive failed reads before attempting reconnect
CAM_RECONNECT_COOLDOWN = 5.0 # min seconds between reconnect attempts, per camera

# ── RECORDING (DVR) ───────────────────────────────────────────────────────────
# Every camera is recorded continuously, WITH what the system drew on it
# (zones, boxes, OK/VIOLATION labels, time), so reviewing footage shows
# exactly what was detected then - nothing is re-analysed on playback, so
# review costs no GPU and can never raise an alert. Frames are JPEGs
# appended to 5-minute segment files plus a small index (time, offset,
# size) per frame; unlike an .mp4 this is readable while it's still being
# written, so even the last few seconds can be rewound. Encoding runs on a
# thread per camera (~1-2 ms a frame). Measured ~0.75 GB/hour/camera;
# old days and anything over the size cap are deleted automatically.
REC_DIR        = "recordings"
REC_W, REC_H   = 640, 480
REC_FPS        = 10
REC_JPEG_Q     = 70
REC_SEGMENT_S  = 300
REC_KEEP_DAYS  = 3      # config "recording_keep_days"
REC_MAX_GB     = 30     # config "recording_max_gb"
REC_IDX        = struct.Struct("<dQI")   # frame time, offset, size
REVIEW_STEP_S  = PLAYBACK_SEEK_SECONDS
LIVE_EDGE_S    = 1.5    # playback this close to now = back to live
ALERT_PREROLL_S = 8     # an alert's playback starts this long before it

def _rec_cam_dir(day, ci):
    return os.path.join(REC_DIR, day, f"cam{ci+1}")

class CamRecorder:
    def __init__(self, ci):
        self.ci = ci
        self.q = queue.Queue(maxsize=REC_FPS * 3)
        self.last_put = 0.0
        self.seg = None          # (start_ts, day, data file, index file)
        threading.Thread(target=self._run, daemon=True).start()

    def wants(self, ts):
        return ts - self.last_put >= 1.0 / REC_FPS

    def offer(self, ts, frame):
        self.last_put = ts
        try: self.q.put_nowait((ts, frame))
        except queue.Full: pass          # disk too slow: drop a frame, never stall the UI

    def close(self):
        self.q.put((None, None))

    def _close_seg(self):
        if self.seg:
            for f in self.seg[2:]:
                try: f.close()
                except Exception: pass
        self.seg = None

    def _open(self, ts):
        self._close_seg()
        d = datetime.fromtimestamp(ts); day = d.strftime("%Y-%m-%d")
        folder = _rec_cam_dir(day, self.ci); os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, d.strftime("%H%M%S"))
        self.seg = (ts, day, open(base + ".jpgs", "ab"), open(base + ".idx", "ab"))

    def _run(self):
        while True:
            ts, frame = self.q.get()
            if ts is None:
                self._close_seg(); return
            try:
                ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, REC_JPEG_Q])
                if not ok: continue
                day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                if self.seg is None or ts - self.seg[0] >= REC_SEGMENT_S or day != self.seg[1]:
                    self._open(ts)
                data_f, idx_f = self.seg[2], self.seg[3]
                data_f.seek(0, 2); off = data_f.tell()
                data_f.write(enc.tobytes()); data_f.flush()
                idx_f.write(REC_IDX.pack(ts, off, len(enc))); idx_f.flush()
            except Exception as e:
                print(f"[RECORDER ERROR] cam{self.ci+1}: {e}"); self._close_seg()

class RecordingReader:
    """Finds the recorded frame for any moment of one camera's footage."""
    def __init__(self, ci):
        self.ci = ci
        self._segs = {}     # day -> (listed_at, [(start_ts, base)])
        self._idx = {}      # base -> [bytes_parsed, [times], [(off, size)]]
        self._last = None; self._last_img = None
        self._spans = None; self._span_cache = {}

    def _seg_span(self, base):
        try: size = os.path.getsize(base + ".idx")
        except OSError: return None
        n = size // REC_IDX.size
        if n == 0: return None
        hit = self._span_cache.get(base)
        if hit and hit[0] == size: return hit[1]
        try:
            with open(base + ".idx", "rb") as f:
                first = REC_IDX.unpack(f.read(REC_IDX.size))[0]
                f.seek((n - 1) * REC_IDX.size); last = REC_IDX.unpack(f.read(REC_IDX.size))[0]
        except (OSError, struct.error):
            return None
        self._span_cache[base] = (size, (first, last))
        return first, last

    def spans(self):
        """[(start, end)] of this camera's recorded footage on all kept
        days, touching segments merged (cached ~2 s) - for the timeline."""
        now = time.time()
        if self._spans and now - self._spans[0] < 2.0: return self._spans[1]
        try: days = sorted(d for d in os.listdir(REC_DIR) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d))
        except OSError: days = []
        out = []
        for day in days:
            for _, base in self._segments(day):
                se = self._seg_span(base)
                if not se: continue
                if out and se[0] - out[-1][1] < 3.0: out[-1] = (out[-1][0], max(out[-1][1], se[1]))
                else: out.append(se)
        self._spans = (now, out)
        return out

    def _segments(self, day):
        hit = self._segs.get(day)
        if hit and time.time() - hit[0] < 2.0: return hit[1]
        segs = []
        folder = _rec_cam_dir(day, self.ci)
        try: names = os.listdir(folder)
        except OSError: names = []
        for fn in names:
            if not fn.endswith(".idx"): continue
            try: st = datetime.strptime(day + fn[:6], "%Y-%m-%d%H%M%S").timestamp()
            except ValueError: continue
            segs.append((st, os.path.join(folder, fn[:-4])))
        segs.sort()
        self._segs[day] = (time.time(), segs)
        return segs

    def _index(self, base):
        ent = self._idx.setdefault(base, [0, [], []])
        try:
            size = os.path.getsize(base + ".idx")
            if size - ent[0] >= REC_IDX.size:            # parse only what was appended
                with open(base + ".idx", "rb") as f:
                    f.seek(ent[0]); raw = f.read(size - ent[0])
                n = len(raw) // REC_IDX.size
                for i in range(n):
                    t, off, sz = REC_IDX.unpack_from(raw, i * REC_IDX.size)
                    ent[1].append(t); ent[2].append((off, sz))
                ent[0] += n * REC_IDX.size
        except OSError:
            pass
        return ent

    def frame_at(self, t):
        """(frame_time, image) of the newest frame at or before t, or
        (None, None) if nothing was recorded within 2 s of t."""
        segs = self._segments(datetime.fromtimestamp(t).strftime("%Y-%m-%d"))
        j = bisect.bisect_right([st for st, _ in segs], t) - 1
        if j < 0: return None, None
        base = segs[j][1]
        _, times, offs = self._index(base)
        k = bisect.bisect_right(times, t) - 1
        if k < 0 or t - times[k] > 2.0: return None, None
        if self._last == (base, k): return times[k], self._last_img
        try:
            with open(base + ".jpgs", "rb") as f:
                f.seek(offs[k][0]); buf = f.read(offs[k][1])
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            return None, None
        self._last = (base, k); self._last_img = img
        return times[k], img

    def has_footage(self, t0, t1):
        t = t0
        while t <= t1:
            if self.frame_at(t)[0] is not None: return True
            t += 1.0
        return False

def rec_cleanup(keep_days, max_gb):
    """Deletes recording days older than keep_days, then the oldest segments
    while the total is over max_gb (never a segment still being written)."""
    if not os.path.isdir(REC_DIR): return
    today = datetime.now().date()
    for d in os.listdir(REC_DIR):
        try: age = (today - datetime.strptime(d, "%Y-%m-%d").date()).days
        except ValueError: continue
        if age >= keep_days:
            shutil.rmtree(os.path.join(REC_DIR, d), ignore_errors=True)
    segs = []
    for root, _, fns in os.walk(REC_DIR):
        for fn in fns:
            if fn.endswith(".jpgs"):
                base = os.path.join(root, fn[:-5])
                try:
                    st = os.stat(base + ".jpgs")
                    size = st.st_size + (os.path.getsize(base + ".idx") if os.path.exists(base + ".idx") else 0)
                    segs.append((st.st_mtime, base, size))
                except OSError:
                    pass
    total = sum(sz for _, _, sz in segs); limit = max_gb * 1e9
    for mtime, base, size in sorted(segs):
        if total <= limit: break
        if time.time() - mtime < 120: continue
        for ext in (".jpgs", ".idx"):
            try: os.remove(base + ext)
            except OSError: pass
        total -= size

_rec_cleanup_started = [False]
def start_rec_cleanup(keep_days, max_gb):
    if _rec_cleanup_started[0]: return
    _rec_cleanup_started[0] = True
    def loop():
        while True:
            try: rec_cleanup(keep_days, max_gb)
            except Exception as e: print(f"[REC CLEANUP ERROR] {e}")
            time.sleep(600)
    threading.Thread(target=loop, daemon=True).start()

def make_rec_frame(proc, ci, cam_name, draws, zone_editor):
    """The frame that gets recorded: analysis frame + zones + detections + time."""
    fr = cv2.resize(proc, (REC_W, REC_H), interpolation=cv2.INTER_LINEAR)
    zone_editor.draw_zones(fr, ci, False, pw=REC_W, ph=REC_H)
    draw_person_overlays(fr, draws, REC_W / PROC_W, REC_H / PROC_H, False)
    # date/time bottom-right (the tile header covers the top when replayed)
    stamp = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    (sw, sh), _ = text_size(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    shade(fr, REC_W-sw-18, REC_H-sh-18, REC_W-4, REC_H-4, THEME["bg_header"], 0.7)
    put_text(fr, stamp, (REC_W-sw-11, REC_H-11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME["text_primary"], 1)
    return fr

def _is_live_source(src):
    try:
        int(src); return True
    except (ValueError, TypeError):
        return not os.path.isfile(str(src))

class LatestFrameCapture:
    """Live cameras (webcam / IP / RTSP) are read on their own thread, keeping
    only the newest frame. Read inline, every loop waited on each camera in
    turn, and a stream that buffers showed up as stutter and growing lag.
    Video files are NOT wrapped - they must play frame by frame."""
    STALE_AFTER = 2.0     # s without a new frame before reads report failure

    def __init__(self, cap):
        self.cap = cap; self._lock = threading.Lock()
        self._frame = None; self._last_ok = time.time(); self._t0 = time.time()
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True); self._th.start()

    def _run(self):
        while self._running:
            try:
                ok, f = self.cap.read()
            except Exception:
                ok, f = False, None
            if ok:
                with self._lock: self._frame = f; self._last_ok = time.time()
            else:
                time.sleep(0.05)

    @property
    def starting(self):
        """Still waiting for the first frame - not a failure yet."""
        return self._frame is None and time.time() - self._t0 < CAM_OPEN_TIMEOUT

    def read(self):
        with self._lock:
            if self._frame is None or time.time() - self._last_ok > self.STALE_AFTER:
                return False, None
            return True, self._frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT: return 0     # live: nothing to seek in
        return self.cap.get(prop)

    def set(self, prop, value):
        return False      # never touch the capture from this thread while _run reads it

    def release(self):
        self._running = False
        self._th.join(timeout=1.0)
        if not self._th.is_alive():      # a read stuck on a dead stream: just leave it
            self.cap.release()

def resolve_source(src):
    """Camera list copied from another PC (e.g. on a USB stick): a video
    file path that doesn't exist here is looked for by its file name in a
    'videos' folder next to the program."""
    s_ = str(src)
    if "://" in s_ or os.path.isfile(s_): return src
    try:
        int(s_); return src
    except ValueError:
        pass
    name = os.path.basename(s_.replace("\\", "/"))
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else BASE_DIR
    for d in (os.path.join(os.getcwd(), "videos"), os.path.join(exe_dir, "videos")):
        if os.path.isfile(os.path.join(d, name)): return os.path.join(d, name)
    return src

def open_capture(src, timeout=CAM_OPEN_TIMEOUT):
    """Opens a VideoCapture on a background thread with a hard timeout, so an
    unreachable IP camera / RTSP stream (bad IP, offline NVR, network drop)
    can never hang the whole app at startup or during a reconnect. The worker
    thread is a daemon: if it's still stuck when the timeout hits, we just
    give up and move on — the thread dies with the process, it won't be
    retried or joined again."""
    src = resolve_source(src)
    result = {"cap": None}
    def worker():
        try:
            cap = cv2.VideoCapture(int(src), cv2.CAP_MSMF)
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(src)
        result["cap"] = cap
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        print(f"[CAMERA] Timed out opening: {src}")
        return None
    cap = result["cap"]
    try:
        if cap is None or not cap.isOpened(): return None
    except Exception:
        return None
    return LatestFrameCapture(cap) if _is_live_source(src) else cap

def open_captures_parallel(cameras):
    """Opens every camera at once instead of one after another — with 3-4
    cameras that each take a second or two (worse for IP/RTSP), sequential
    opening made the whole app look frozen for several seconds right after
    clicking Launch, before the window even appeared. Each open_capture()
    call already runs its own watchdog thread internally, so running them
    from N parallel threads here turns total wait time from sum(...) into
    roughly max(...)."""
    caps = [None]*len(cameras)
    def worker(i, src):
        caps[i] = open_capture(src)
    threads = [threading.Thread(target=worker, args=(i, cam["source"]), daemon=True)
               for i, cam in enumerate(cameras)]
    for th in threads: th.start()
    for th in threads: th.join(CAM_OPEN_TIMEOUT + 2)
    return caps

def run_monitor(cfg, person_model):
    person_det = PersonDetector(person_model)
    global TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_TOKEN  =cfg.get("telegram_token","")
    TELEGRAM_CHAT_ID=cfg.get("telegram_chat_id","")

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        threading.Thread(target=_telegram_check_worker,
                          args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID), daemon=True).start()

    if cfg.get("remote_enabled"):
        start_remote_access(cfg)

    # Windows' default 15.6 ms timer tick rounds every short waitKey() up to
    # 16/31 ms, which alone cost ~10 fps; 1 ms resolution for this process
    # (what video players do) lets the frame pacing below actually work.
    if sys.platform == "win32":
        try:
            import ctypes; ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    cameras=cfg["cameras"]
    if not cameras: return False

    n=len(cameras)
    if   n<=1: cols=1; base_pw=960   # single camera gets a genuinely large view
    elif n<=4: cols=2; base_pw=640
    elif n<=9: cols=3; base_pw=480
    else:      cols=4; base_pw=400
    rows=(n+cols-1)//cols
    # PW/PH are re-derived from the actual OS window size every frame below
    # (search MIN_TILE_W) so the whole UI stays crisp — never stretched — at
    # any window size the user drags to. These starting values just pick a
    # sane initial window size before the first resize event.
    PW=base_pw; PH=int(PW*3/4)
    grid_w=cols*PW

    zone_editor=ZoneEditor(); zone_editor.load(cfg)
    room_watch=RoomWatch(cfg)
    ZONES_PAUSE["off"]=cfg.get("zones_paused")

    WIN=APP_NAME
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, grid_w+PANEL_W, TOP_BAR_H+rows*PH+PLAYBACK_BAR_H)
    # NOTE: setting the OpenCV/taskbar icon here (immediately after
    # namedWindow) is unreliable - FindWindowW can fail to locate the
    # window before it's actually mapped on screen. The real call happens
    # once, after the first cv2.imshow() below (search for _icon_applied).
    _icon_applied = False

    # Show *something* immediately — opening cameras (especially IP/RTSP)
    # can take a few seconds each; doing that before the window ever
    # appears is exactly what made the app look frozen right after
    # clicking Launch. Cameras below open in parallel, not one by one.
    loading = np.full((TOP_BAR_H+rows*PH+PLAYBACK_BAR_H, grid_w+PANEL_W, 3), THEME["bg_panel"], dtype=np.uint8)
    text_centered(loading, "Opening cameras...", (grid_w+PANEL_W)//2,
                  (TOP_BAR_H+rows*PH+PLAYBACK_BAR_H)//2, 1.0, THEME["accent"], 2)
    cv2.imshow(WIN, loading)
    cv2.waitKey(1)

    caps = open_captures_parallel(cameras)
    for cap, cam in zip(caps, cameras):
        if cap is None:
            add_log(f"Could not open {cam['name']} — check source/network", THEME["danger"])
    cam_fail_count      = defaultdict(int)
    cam_last_reconnect  = defaultdict(float)

    fps=30
    for cap in caps:
        if cap:
            f=cap.get(cv2.CAP_PROP_FPS)
            if f>0: fps=f; break
    frame_delay=max(1,int(1000/fps))

    edit_mode=False; drawing_zone=False
    # What the next drawn zone will be (set in the Select panel), and which
    # existing zone - if any - that same panel is currently editing.
    new_zone_kind="ppe_check"; new_zone_checks=set(DEFAULT_CHECKS)
    editing_zone=None      # (ci, zi) or None
    zone_menu_visible=False; active_menu_cam=None; zone_menu_pos=(0,0)
    new_zone_pts=[]; active_draw_cam=0
    quit_requested=False; back_requested=False
    active_panel=None
    selected_cam=0
    focus_cam=None               # camera shown large (double-click), or None = grid
    last_click=[0.0,None]        # (time, camera) of the last click, for double-click
    cam_overlays={}              # ci -> (draws, unchecked) from process_cam_frame
    reviews={}                   # ci -> {"t","paused","tick","alert"}: tile shows recorded footage
    readers=[RecordingReader(i) for i in range(n)]
    rec_on=cfg.get("recording_enabled", True)
    recorders=[CamRecorder(i) for i in range(n)] if rec_on else [None]*n
    if rec_on:
        start_rec_cleanup(cfg.get("recording_keep_days", REC_KEEP_DAYS), cfg.get("recording_max_gb", REC_MAX_GB))
    logui={"tab":"log", "day":datetime.now().strftime("%Y-%m-%d"), "scroll":0, "sel":None, "click":(0.0,None)}
    last_zone_expiry=[0.0]
    last_frames=[None]*n

    layout={"cols":cols,"rows":rows,"PW":PW,"PH":PH,"grid_w":grid_w,"grid_h":rows*PH,
            "total_w":grid_w+PANEL_W, "icon_rects":{}, "panel_hits":[], "playback_rects":{},
            "rects":compute_tiles(n,cols,rows,PW,PH,None,4/3)[0], "hover":None, "frames":[]}

    def toggle_focus(ci):
        nonlocal focus_cam, selected_cam
        if n < 2: return
        focus_cam = None if focus_cam == ci else ci
        selected_cam = ci

    def start_review(ci, t, paused=False, alert=False):
        if not rec_on:
            add_log("Recording is off - nothing to rewind", THEME["warn"]); return
        now=time.time()
        reviews[ci]={"t":min(t,now), "paused":paused, "tick":now, "alert":alert, "speed":1}

    def review_cmd(key, ci=None):
        """Rewind / forward / pause / live for one camera's tile. Only what
        the tile SHOWS changes - its live picture keeps being analysed."""
        ci = selected_cam if ci is None else ci
        rv = reviews.get(ci); now = time.time()
        if key=="rewind":
            if rv: rv["t"] -= REVIEW_STEP_S; rv["alert"] = False
            else:  start_review(ci, now - REVIEW_STEP_S)
        elif key=="forward":
            if rv:
                rv["t"] += REVIEW_STEP_S; rv["alert"] = False
                if rv["t"] >= now - LIVE_EDGE_S: reviews.pop(ci, None)
        elif key=="pauseplay":
            if rv is None:  start_review(ci, now, paused=True)
            elif rv["paused"] or rv["speed"] != 1:      # play = normal speed
                rv["paused"] = False; rv["speed"] = 1; rv["tick"] = now
            else:           rv["paused"] = True
        elif key=="ffwd":
            if rv:       # nothing to fast-forward into while already live
                sp = rv["speed"]
                rv["speed"] = next((v for v in REVIEW_SPEEDS if v > sp), REVIEW_SPEEDS[0]) if sp >= 1 else REVIEW_SPEEDS[0]
                rv["paused"] = False; rv["tick"] = now
        elif key=="frew":
            if rv is None:
                start_review(ci, now); rv = reviews.get(ci)
            if rv:
                sp = -rv["speed"]
                rv["speed"] = -(next((v for v in REVIEW_SPEEDS if v > sp), REVIEW_SPEEDS[0]) if sp >= 1 else REVIEW_SPEEDS[0])
                rv["paused"] = False; rv["tick"] = now
        elif key=="live":
            reviews.pop(ci, None)

    def seek_to(t):
        """Timeline click/drag: jump the selected camera's tile to time t."""
        ci = selected_cam; now = time.time()
        if t >= now - LIVE_EDGE_S:
            reviews.pop(ci, None); return
        sp_ = readers[ci].spans()
        if sp_ and t < sp_[0][0]: t = sp_[0][0]
        rv = reviews.get(ci)
        if rv: rv["t"] = t; rv["alert"] = False; rv["tick"] = now
        else:  start_review(ci, t)

    def open_alert(entry):
        nonlocal focus_cam, selected_cam
        ci = entry.get("cam", -1)
        if not 0 <= ci < n:
            add_log("That alert's camera is no longer set up", THEME["warn"]); return
        t0 = entry["ts"] - ALERT_PREROLL_S
        if not rec_on or not readers[ci].has_footage(t0, entry["ts"] + 2):
            add_log("No recording kept for that alert", THEME["warn"]); return
        start_review(ci, t0, alert=True)
        selected_cam = ci
        if n >= 2: focus_cam = ci

    def take_snapshot():
        fr = layout["frames"]
        if not (0 <= selected_cam < len(fr)) or fr[selected_cam] is None: return
        try:
            os.makedirs("snapshots", exist_ok=True)
            fname = f"snapshots/{cameras[selected_cam]['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            ok, enc = cv2.imencode(".jpg", fr[selected_cam])   # imwrite can't do non-ASCII names
            if ok:
                with open(fname, "wb") as f: f.write(enc.tobytes())
            add_log(f"Snapshot saved: {fname}", THEME["accent"])
        except Exception as e:
            add_log(f"Snapshot failed: {e}", THEME["danger"])

    def editing_zone_obj():
        if editing_zone is None: return None
        eci, ezi = editing_zone
        zs = zone_editor.get_zones(eci)
        return zs[ezi] if 0 <= ezi < len(zs) else None

    def shift_action(action, value):
        s = room_watch.s
        if action=="toggle":
            s["enabled"] = not s["enabled"]
            if not s["enabled"]: room_watch.simulate = False
            add_log(f"After-hours alerts {'ON' if s['enabled'] else 'OFF'}", THEME["accent"])
        elif action in ("start","end"):
            key = "shift_"+action
            s[key] = _min_to_hm(_hm_to_min(s[key]) + value)
        elif action=="grace":
            s["grace_min"] = max(0, min(120, int(s.get("grace_min",0)) + value))
        elif action=="mode":
            s["mode"] = value
            add_log("After hours, alert if: " + dict(ROOM_MODES)[value], THEME["accent"])
        elif action=="cam":
            s["off_cams"] = sorted(set(s.get("off_cams",[])) ^ {value})
        elif action=="calib":
            if not room_watch.calibrate(selected_cam, value):
                add_log("No picture from this camera yet", THEME["warn"])
            elif value=="reset":
                add_log(f"{cameras[selected_cam]['name']}: light level reset to default", THEME["accent"])
            else:
                add_log(f"{cameras[selected_cam]['name']}: learned 'lights {value.upper()}' level "
                        f"({room_watch.s['calib'][str(selected_cam)][value]:.0f})", THEME["accent"])
        elif action=="test":
            room_watch.simulate = not room_watch.simulate
            if room_watch.simulate and not s["enabled"]:
                s["enabled"] = True
                add_log("After-hours alerts ON", THEME["accent"])
            add_log("After-hours TEST " + ("started" if room_watch.simulate else "stopped"), THEME["warn"])
        room_watch.save()

    def mouse_cb(event,x,y,flags,param):
        nonlocal drawing_zone,new_zone_pts,active_draw_cam,new_zone_kind,editing_zone
        nonlocal zone_menu_visible,active_menu_cam,zone_menu_pos
        nonlocal quit_requested,back_requested,active_panel,selected_cam,edit_mode

        gw=layout["grid_w"]; pw=layout["PW"]; ph=layout["PH"]

        # hover - for the bar buttons' highlight + tooltip
        hov = None
        if y < TOP_BAR_H:
            hov = next((k for k,(rx1,ry1,rx2,ry2) in layout["icon_rects"].items()
                        if rx1<=x<=rx2 and ry1<=y<=ry2), None)
        elif y >= TOP_BAR_H + layout["grid_h"]:
            pby = y - (TOP_BAR_H + layout["grid_h"])
            hov = next((("pb",k) for k,(rx1,ry1,rx2,ry2) in layout["playback_rects"].items()
                        if rx1<=x<=rx2 and ry1<=pby<=ry2), None)
        layout["hover"] = hov; layout["mouse"] = (x, y)

        # timeline scrubbing: keeps following the mouse while the button is held
        if layout.get("scrub"):
            if event==cv2.EVENT_LBUTTONUP:
                layout["scrub"]=False; return
            if event==cv2.EVENT_MOUSEMOVE and layout.get("timeline"):
                t0,t1,tx1,tx2 = layout["timeline"]
                seek_to(t0 + (t1-t0)*min(1.0,max(0.0,(x-tx1)/max(1,tx2-tx1)))); return

        # 1) top bar icon clicks
        if y < TOP_BAR_H:
            if event==cv2.EVENT_LBUTTONDOWN:
                for key,(rx1,ry1,rx2,ry2) in layout["icon_rects"].items():
                    if rx1<=x<=rx2 and ry1<=y<=ry2:
                        if key=="back":
                            back_requested=True
                        elif key=="edit":
                            edit_mode=not edit_mode; drawing_zone=False; new_zone_pts=[]
                            zone_menu_visible=False; zone_editor.deselect()
                            add_log(f"Edit mode {'ON' if edit_mode else 'OFF'}",THEME["accent"])
                        elif key=="pauseplay":
                            review_cmd("pauseplay")
                        else:
                            active_panel = None if active_panel==key else key
                        return
            return

        # 1b) review bar — rewind/pause/forward/stop, always acts on
        # selected_cam regardless of where in the bar's width it's clicked
        # (it spans the full window width like the top bar does)
        if y >= TOP_BAR_H + layout["grid_h"]:
            if event==cv2.EVENT_LBUTTONDOWN:
                # playback_rects are local to the review bar's own canvas
                # (y in [0, PLAYBACK_BAR_H)), same convention as panel_hits
                # being local to the sidebar - offset the real y before testing
                pby = y - (TOP_BAR_H + layout["grid_h"])
                for key,(rx1,ry1,rx2,ry2) in layout["playback_rects"].items():
                    if rx1<=x<=rx2 and ry1<=pby<=ry2:
                        if key=="snap":  take_snapshot(); return
                        if key=="focus": toggle_focus(focus_cam if focus_cam is not None else selected_cam); return
                        if key=="timeline" and layout.get("timeline"):
                            t0,t1,tx1,tx2 = layout["timeline"]
                            layout["scrub"]=True
                            seek_to(t0 + (t1-t0)*min(1.0,max(0.0,(x-tx1)/max(1,tx2-tx1))))
                        if key in ("rewind","forward","pauseplay","live","ffwd","frew"):
                            review_cmd(key)
                        return
            return

        # 2) fixed sidebar — empty and inert when no panel is chosen, else
        # handles clicks in its own content; a click on the empty sidebar
        # does nothing (there's nothing to click), it just doesn't fall
        # through to the camera grid underneath it
        if x >= gw:
            if active_panel:
                if event==cv2.EVENT_LBUTTONDOWN:
                    lx,ly = x-gw, y-TOP_BAR_H
                    for (rx1,ry1,rx2,ry2,payload) in layout["panel_hits"]:
                        if rx1<=lx<=rx2 and ry1<=ly<=ry2:
                            if active_panel=="select":
                                action, value = payload
                                target = editing_zone_obj()
                                if action=="kind":
                                    if target is not None:
                                        zone_editor.set_zone_settings(target, ztype=value)
                                        zone_editor.save_to(cfg); save_config(cfg)
                                    else:
                                        new_zone_kind = value
                                elif action=="item":
                                    if target is not None:
                                        checks = set(target.get("checks", [])) ^ {value}
                                        if not checks:
                                            add_log("A zone needs at least one check", THEME["warn"])
                                        else:
                                            zone_editor.set_zone_settings(target, checks=checks)
                                            zone_editor.save_to(cfg); save_config(cfg)
                                    else:
                                        new_zone_checks.symmetric_difference_update({value})
                                elif action=="draw":
                                    drawing_zone=True; new_zone_pts=[]; active_panel=None
                                    preview = zone_auto_name(new_zone_kind, new_zone_checks)
                                    add_log(f"Drawing [{preview}] - left=add, right=finish", THEME["ok"])
                                elif action in ("off","on") and target is not None:
                                    if action=="on":
                                        zone_editor.set_zone_off(target, None)
                                        add_log(f"Zone '{target['name']}' turned ON", THEME["ok"])
                                    elif value=="manual":
                                        zone_editor.set_zone_off(target, "manual")
                                        add_log(f"Zone '{target['name']}' OFF until turned on", THEME["warn"])
                                    else:
                                        zone_editor.set_zone_off(target, time.time()+value*3600)
                                        add_log(f"Zone '{target['name']}' {zone_off_text(target)}", THEME["warn"])
                                    zone_editor.save_to(cfg); save_config(cfg)
                                elif action=="done":
                                    if target is not None:
                                        add_log(f"Zone updated: {target['name']}", THEME["accent"])
                                    editing_zone=None; zone_editor.deselect()
                            elif active_panel=="shift":
                                shift_action(*payload)
                            elif active_panel=="power":
                                action, value = payload
                                if action=="resume":
                                    ZONES_PAUSE["off"]=None
                                    add_log("ALL zones turned back ON", THEME["ok"]); log_file("ZONES_RESUMED")
                                else:
                                    ZONES_PAUSE["off"]="manual" if value=="manual" else time.time()+value*3600
                                    cam_tracked.clear(); cam_nid.clear()     # nobody stays "armed" mid-check
                                    add_log(f"ALL zones OFF {zones_pause_text()}", THEME["warn"])
                                    log_file("ZONES_PAUSED", zones_pause_text())
                                cfg["zones_paused"]=ZONES_PAUSE["off"]; save_config(cfg)
                            elif active_panel=="log":
                                action, value = payload
                                if action=="tab":
                                    logui["tab"]=value; logui["scroll"]=0
                                elif action=="day":
                                    days=alert_days()
                                    i=days.index(logui["day"]) if logui["day"] in days else len(days)-1
                                    logui["day"]=days[max(0,min(len(days)-1,i+value))]; logui["scroll"]=0
                                elif action=="alert":
                                    t0,prev=logui["click"]
                                    if prev is value and time.time()-t0 < DOUBLE_CLICK_S:
                                        logui["click"]=(0.0,None); open_alert(value)
                                    else:
                                        logui["click"]=(time.time(),value); logui["sel"]=value
                            elif active_panel=="tools":
                                ci,zi = payload
                                zone_editor.select_zone_by_index(ci,zi)
                                editing_zone=(ci,zi); active_panel="select"
                            return
                elif event==cv2.EVENT_MOUSEWHEEL and active_panel=="log" and logui["tab"]=="alerts":
                    logui["scroll"]=max(0,logui["scroll"]+(-1 if flags>0 else 1))
                elif event==cv2.EVENT_MOUSEWHEEL and active_panel=="log":
                    max_scroll=max(0,len(admin_log)-1)
                    if flags>0: log_scroll_pos[0]=min(log_scroll_pos[0]+3,max_scroll)
                    else:       log_scroll_pos[0]=max(log_scroll_pos[0]-3,0)
            return

        if event==cv2.EVENT_LBUTTONDOWN and zone_menu_visible:
            mx1,my1=zone_menu_pos
            mx2=mx1+MENU_W; my2=my1+MENU_PAD*2+MENU_ITEM_H*len(ZONE_MENU_ITEMS)
            if mx1<=x<=mx2 and my1<=y<=my2:
                idx=zone_menu_hit(mx1,my1,x,y)
                if idx is not None and active_menu_cam is not None:
                    label=ZONE_MENU_ITEMS[idx][0]
                    if   label=="Edit Checks":
                        if zone_editor.selected_zone is not None:
                            editing_zone=(active_menu_cam, zone_editor.selected_zone)
                            active_panel="select"
                    elif label=="Turn Off / On":
                        zs = zone_editor.get_zones(active_menu_cam); zi = zone_editor.selected_zone
                        if zi is not None and 0 <= zi < len(zs):
                            z = zs[zi]
                            zone_editor.set_zone_off(z, None if not zone_on(z) else "manual")
                            add_log(f"Zone '{z['name']}' turned {'ON' if zone_on(z) else 'OFF'}",
                                    THEME["ok"] if zone_on(z) else THEME["warn"])
                    elif label=="Zoom In":     zone_editor.resize_selected(active_menu_cam,1.1)
                    elif label=="Zoom Out":    zone_editor.resize_selected(active_menu_cam,0.9)
                    elif label=="Delete Zone":
                        zone_editor.delete_selected(active_menu_cam)
                        editing_zone=None   # indexes shift after a delete
                        add_log("Zone deleted",THEME["danger"])
                    elif label=="Exit Program": quit_requested=True
                    zone_editor.save_to(cfg); save_config(cfg)
                zone_menu_visible=False; return
            else:
                zone_menu_visible=False; return

        # 3) camera tiles - the grid, or one enlarged camera + the strip
        gy = y - TOP_BAR_H
        hit = next(((i, x-tx, gy-ty, tw, th) for i,(tx,ty,tw,th) in enumerate(layout["rects"])
                    if tx<=x<tx+tw and ty<=gy<ty+th), None)
        if hit is None: return
        ci,cx,cy,pw,ph = hit
        if focus_cam is not None and ci!=focus_cam and (drawing_zone or edit_mode):
            return      # while enlarged, zones are drawn/edited on the big camera only
        if not drawing_zone and not edit_mode:
            # OpenCV may deliver the 2nd click as DBLCLK or as a plain DOWN
            # depending on the window class, so both are handled.
            if event==cv2.EVENT_LBUTTONDBLCLK or (event==cv2.EVENT_LBUTTONDOWN and last_click[1]==ci
                                                   and time.time()-last_click[0] < DOUBLE_CLICK_S):
                toggle_focus(ci); last_click[:]=[0.0,None]; return
            if event==cv2.EVENT_LBUTTONDOWN:
                last_click[:]=[time.time(),ci]

        if drawing_zone:
            if event==cv2.EVENT_LBUTTONDOWN:
                # stored in the fixed virtual space, not current screen pixels,
                # so the zone stays aligned however the window gets resized later
                active_draw_cam=ci
                new_zone_pts.append([int(cx*ZONE_BASE_W/pw), int(cy*ZONE_BASE_H/ph)])
            elif event==cv2.EVENT_RBUTTONDOWN:
                if len(new_zone_pts)>=3:
                    zone_editor.add_zone(active_draw_cam,new_zone_kind,new_zone_pts.copy(),
                                         checks=new_zone_checks)
                    zone_editor.save_to(cfg); save_config(cfg)
                    add_log(f"Zone saved on CAM{active_draw_cam+1}: "
                            f"{zone_editor.get_zones(active_draw_cam)[-1]['name']}",THEME["accent"])
                drawing_zone=False; new_zone_pts=[]
        elif edit_mode:
            zone_editor.handle_mouse(event,cx,cy,ci,pw=pw,ph=ph)
            if event==cv2.EVENT_RBUTTONDOWN and zone_editor.selected_zone is not None:
                zone_menu_visible=True; active_menu_cam=ci; zone_menu_pos=(x,y)
            if event==cv2.EVENT_LBUTTONUP:
                zone_editor.save_to(cfg); save_config(cfg)
        elif event==cv2.EVENT_LBUTTONDOWN:
            selected_cam=ci

    cv2.setMouseCallback(WIN,mouse_cb)
    add_log("System started",THEME["accent"])
    add_log("Use Select (top bar) to pick what a zone checks, then draw it",THEME["text_secondary"])
    missing_now = set(CHECK_ITEMS) - supported_items()
    if missing_now:
        add_log("No model yet for: " + ", ".join(CHECK_ITEMS[i]["label"] for i in ordered_checks(missing_now)),
                THEME["warn"])
    print("Running - Q=quit  E=edit  R=reset  B=back  SPACE=pause selected  S=snapshot  Up/Down=scroll log")

    MIN_TILE_W, MIN_TILE_H = 160, 120
    while True:
        if quit_requested or back_requested: break
        now_ts=time.time()
        room_watch.tick()
        if now_ts-last_zone_expiry[0] >= 1.0:       # "off for N hours" zones coming back on
            last_zone_expiry[0]=now_ts
            back=zone_editor.expire_offs(now_ts)
            for zci,z in back:
                add_log(f"Zone '{z['name']}' on CAM{zci+1} is back ON", THEME["ok"])
            if back: zone_editor.save_to(cfg); save_config(cfg)
            off_=ZONES_PAUSE["off"]
            if off_ not in (None,"manual") and now_ts >= off_:
                ZONES_PAUSE["off"]=None; cfg["zones_paused"]=None; save_config(cfg)
                add_log("ALL zones are back ON", THEME["ok"]); log_file("ZONES_RESUMED")

        # Re-render everything at whatever size the OS window actually is
        # right now, instead of stretching a fixed-size bitmap to fit — that
        # stretch is what caused the blurry text/icons when enlarging.
        try:
            _,_,win_w,win_h = cv2.getWindowImageRect(WIN)
            if win_w>0 and win_h>0:
                PW = max(MIN_TILE_W, (win_w-PANEL_W)//cols)
                PH = max(MIN_TILE_H, (win_h-TOP_BAR_H-PLAYBACK_BAR_H)//rows)
        except Exception:
            pass
        layout["PW"]=PW; layout["PH"]=PH

        # The zone editor only lives inside the Select panel; leaving the
        # panel (or the zone disappearing) ends the edit session.
        if active_panel != "select" or editing_zone_obj() is None:
            editing_zone = None

        # Tile layout for this frame: the normal grid, or - after a double-
        # click - one camera large with the others in a strip underneath.
        if focus_cam is not None and (focus_cam >= n or n < 2): focus_cam = None
        f_aspect = 4/3
        if focus_cam is not None and last_frames[focus_cam] is not None:
            fh_, fw_ = last_frames[focus_cam].shape[:2]; f_aspect = fw_/max(1, fh_)
        rects, grid_w, grid_h, strip_y = compute_tiles(n, cols, rows, PW, PH, focus_cam, f_aspect)
        layout["rects"]=rects; layout["grid_w"]=grid_w; layout["grid_h"]=grid_h
        grid=np.full((grid_h,grid_w,3),THEME["bg_panel"],dtype=np.uint8)
        if strip_y is not None:
            cv2.rectangle(grid,(0,strip_y),(grid_w,grid_h),THEME["bg_header"],-1)
            cv2.line(grid,(0,strip_y),(grid_w,strip_y),THEME["border"],1)
        else:
            for slot in range(n, cols*rows):
                sx_, sy_ = (slot % cols)*PW, (slot // cols)*PH
                text_centered(grid,"NO CAMERA",sx_+PW//2,sy_+PH//2,0.6,THEME["text_dim"],1)
        frames=[None]*n; layout["frames"]=frames
        try:
            for ci,(cap,cam) in enumerate(zip(caps,cameras)):
                tx,ty,tw,th = rects[ci]
                compact = th < COMPACT_TILE_H
                # The live picture is ALWAYS read, analysed and recorded -
                # also while this camera's tile is showing recorded footage.
                raw=None
                if cap is None:
                    if now_ts-cam_last_reconnect[ci] > CAM_RECONNECT_COOLDOWN:
                        cam_last_reconnect[ci]=now_ts
                        cap=open_capture(cam["source"],timeout=3.0)
                        caps[ci]=cap
                        if cap is not None:
                            add_log(f"{cam['name']} reconnected",THEME["ok"])
                            cam_fail_count[ci]=0
                if cap is not None:
                    starting = getattr(cap,"starting",False)
                    ret,frame=cap.read()
                    if not ret and not starting:     # video file: loop back to the start
                        cap.set(cv2.CAP_PROP_POS_FRAMES,0); ret,frame=cap.read()
                    if ret:
                        cam_fail_count[ci]=0
                        raw=frame; last_frames[ci]=frame   # never drawn on - safe to keep
                    elif not starting:
                        cam_fail_count[ci]+=1
                        if (cam_fail_count[ci]>=CAM_RECONNECT_AFTER
                                and now_ts-cam_last_reconnect[ci]>CAM_RECONNECT_COOLDOWN):
                            cam_last_reconnect[ci]=now_ts
                            cap.release()
                            new_cap=open_capture(cam["source"],timeout=3.0)
                            caps[ci]=new_cap
                            if new_cap is not None:
                                add_log(f"{cam['name']} reconnected",THEME["ok"])
                                cam_fail_count[ci]=0

                if raw is not None:
                    # analysed at the fixed PROC size, shown at the tile size
                    proc=cv2.resize(raw,(PROC_W,PROC_H),interpolation=cv2.INTER_AREA)
                    room_watch.measure(ci,proc,now_ts)
                    person_det.submit(ci,proc,now_ts,raw)       # analysed on the person thread
                    det=person_det.get(ci)
                    cam_overlays[ci]=process_cam_frame(proc,ci,zone_editor,now_ts,det)
                    room_watch.update(ci,proc,len(det[0]) if det else 0,now_ts,cam["name"])
                    rec=recorders[ci]
                    if rec is not None and rec.wants(now_ts):
                        rec.offer(now_ts, make_rec_frame(proc, ci, cam["name"], cam_overlays[ci][0], zone_editor))
                else:
                    room_watch.status.pop(ci,None)

                rv=reviews.get(ci)
                if rv is not None:
                    if not rv["paused"]: rv["t"]+=(now_ts-rv["tick"])*rv.get("speed",1)
                    rv["tick"]=now_ts; rv["t"]=min(rv["t"],now_ts)
                    if rv.get("speed",1) < 0 and not rv["paused"]:
                        sp_ = readers[ci].spans()
                        if sp_ and rv["t"] <= sp_[0][0]:      # rewound to the very start
                            rv["t"]=sp_[0][0]; rv["paused"]=True; rv["speed"]=1
                    if not rv["paused"] and rv["t"]>=now_ts-LIVE_EDGE_S:
                        reviews.pop(ci,None); rv=None        # caught up with live

                if rv is not None:
                    _,rimg = readers[ci].frame_at(rv["t"])
                    if rimg is None and not rv["paused"]:     # a gap (program was off): skip it
                        sp_ = readers[ci].spans()
                        if rv.get("speed",1) > 0:
                            nxt = next((a for a,b in sp_ if a > rv["t"]), None)
                            if nxt is None: reviews.pop(ci,None); rv=None
                            else: rv["t"]=nxt
                        else:
                            prv = next((b for a,b in reversed(sp_) if b < rv["t"]), None)
                            if prv is not None: rv["t"]=prv
                        if rv is not None: _,rimg = readers[ci].frame_at(rv["t"])
                    if rimg is not None:
                        shrink = tw*th < rimg.shape[0]*rimg.shape[1]
                        tile=cv2.resize(rimg,(tw,th),interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
                    else:
                        tile=np.full((th,tw,3),THEME["bg_panel"],dtype=np.uint8)
                        text_centered(tile,"NO RECORDING AT THIS TIME",tw//2,th//2+6,0.38 if compact else 0.6,
                                      THEME["text_dim"],1)
                elif raw is None:
                    tile=np.full((th,tw,3),THEME["bg_panel"],dtype=np.uint8)
                    starting = caps[ci] is not None and getattr(caps[ci],"starting",False)
                    msg = "CONNECTING..." if starting else "NO SIGNAL"
                    text_centered(tile,msg,tw//2,th//2+6,0.45 if compact else 0.8,
                                  THEME["warn"] if starting else THEME["danger"],1 if compact else 2)
                else:
                    shrink = tw*th < raw.shape[0]*raw.shape[1]
                    tile=cv2.resize(raw,(tw,th),interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
                    draws,unchecked = cam_overlays.get(ci,([],set()))
                    draw_person_overlays(tile,draws,tw/PROC_W,th/PROC_H,compact)
                    if not compact: draw_model_warning(tile,unchecked)
                    if zones_paused(now_ts) and zone_editor.get_zones(ci):
                        msg_ = "ALL ZONES OFF " + zones_pause_text()
                        if compact:
                            shade(tile,0,tile.shape[0]-18,tw,tile.shape[0],THEME["warn"],0.85)
                            put_text(tile,fit_px(msg_,tw-8,0.28),(4,tile.shape[0]-5),cv2.FONT_HERSHEY_SIMPLEX,0.28,THEME["black"],1)
                        else:
                            shade(tile,0,30,tw,54,THEME["warn"],0.85)
                            draw_glyph_icon(tile,"power",16,42,14,THEME["black"])
                            put_text(tile,msg_,(30,47),cv2.FONT_HERSHEY_SIMPLEX,0.38,THEME["black"],1)

                editable = (focus_cam is None or ci == focus_cam) and rv is None
                if rv is None:            # recorded footage already has its zones burned in
                    zone_editor.draw_zones(tile,ci,edit_mode and editable,pw=tw,ph=th)
                    if editing_zone is not None and editing_zone[0]==ci:
                        ez_pts = zone_editor.scaled_points(editing_zone_obj(), tw, th)
                        if len(ez_pts) >= 3:
                            cv2.polylines(tile,[ez_pts],True,THEME["white"],3)
                    if drawing_zone and active_draw_cam==ci and new_zone_pts:
                        col=ZONE_TYPES[new_zone_kind]["color"]
                        disp_pts=[[int(px*tw/ZONE_BASE_W),int(py*th/ZONE_BASE_H)] for px,py in new_zone_pts]
                        for pt in disp_pts: cv2.circle(tile,tuple(pt),5,col,-1)
                        if len(disp_pts)>1:
                            cv2.polylines(tile,[np.array(disp_pts,dtype=np.int32)],False,col,1)

                # header: translucent strip with the camera name (+ REC dot)
                hh = 22 if compact else 30
                shade(tile,0,0,tw,hh,THEME["bg_header"],0.78)
                draw_glyph_icon(tile,"video",13 if compact else 17,hh//2,13 if compact else 17,
                                THEME["warn"] if rv is not None else THEME["accent"])
                put_text(tile,_fit(cam["name"],max(4,tw//9)),(26 if compact else 34,hh-7 if compact else hh-10),
                         cv2.FONT_HERSHEY_SIMPLEX,0.34 if compact else 0.48,THEME["text_primary"],1 if compact else 2)
                badge_x2 = tw-6
                if edit_mode and editable and not compact:
                    draw_edit_badge(tile); badge_x2 = tw-52
                if rec_on and rv is None and raw is not None and not compact:
                    cv2.circle(tile,(badge_x2-34,hh//2),4,THEME["danger"],-1,cv2.LINE_AA)
                    put_text(tile,"REC",(badge_x2-26,hh//2+4),cv2.FONT_HERSHEY_SIMPLEX,0.3,THEME["text_secondary"],1)
                if ci==selected_cam:
                    cv2.rectangle(tile,(1,1),(tw-2,th-2),THEME["warn"] if rv is not None else THEME["accent"],
                                  2 if compact else 3)
                elif rv is not None:
                    cv2.rectangle(tile,(1,1),(tw-2,th-2),THEME["warn"],1)

                if rv is not None:
                    draw_review_badge(tile, rv, now_ts, compact)
                elif not compact:
                    # timestamp burn-in, bottom-right corner — this machine's own
                    # clock, same on every tile (matches real CCTV overlay style)
                    ts_text = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                    (ts_w, ts_h), _ = text_size(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    ts_x2, ts_y2 = tw-6, th-6
                    ts_x1, ts_y1 = ts_x2-ts_w-14, ts_y2-ts_h-12
                    shade(tile,ts_x1,ts_y1,ts_x2,ts_y2,THEME["bg_header"],0.75)
                    put_text(tile,ts_text,(ts_x1+7,ts_y2-7),cv2.FONT_HERSHEY_SIMPLEX,0.4,
                             THEME["text_primary"],1)
                    draw_room_badge(tile, room_watch.status.get(ci))
                else:
                    rs = room_watch.status.get(ci)
                    if rs and rs.get("state")=="alarm":
                        cv2.circle(tile,(tw-10,hh//2),5,THEME["danger"],-1,cv2.LINE_AA)

                frames[ci]=tile
                grid[ty:ty+th,tx:tx+tw]=tile
        except Exception as e:
            print(f"[FRAME LOOP ERROR] {e}")
            add_log(f"Frame processing error: {e}", THEME["danger"])

        # fixed docked sidebar — empty when idle, filled with the active
        # panel's content otherwise; always PANEL_W wide, never eats into
        # the camera grid's own width
        panel_hits=[]
        if active_panel=="log":
            panel_canvas,panel_hits = build_panel_log(PANEL_W, grid_h, logui)
        elif active_panel=="select":
            ez = editing_zone_obj()
            if ez is not None:
                panel_canvas,panel_hits = build_panel_select(
                    PANEL_W, grid_h, ez["type"], set(ez.get("checks", [])),
                    f"{cameras[editing_zone[0]]['name']} · {ez['name']}", supported_items(), off_zone=ez)
            else:
                panel_canvas,panel_hits = build_panel_select(
                    PANEL_W, grid_h, new_zone_kind, new_zone_checks, None, supported_items())
        elif active_panel=="tools":
            panel_canvas,panel_hits = build_panel_tools(PANEL_W, grid_h, zone_editor, cameras,
                                                         supported_items())
        elif active_panel=="shift":
            panel_canvas,panel_hits = build_panel_shift(PANEL_W, grid_h, room_watch, cameras, selected_cam)
        elif active_panel=="power":
            panel_canvas,panel_hits = build_panel_power(PANEL_W, grid_h, zone_editor)
        elif active_panel=="help":
            panel_canvas,panel_hits = build_panel_help(PANEL_W, grid_h)
        else:
            panel_canvas = build_panel_empty(PANEL_W, grid_h)
        layout["panel_hits"]=panel_hits

        body=np.hstack([grid, panel_canvas])
        total_w=body.shape[1]; layout["total_w"]=total_w

        hov=layout["hover"]
        sel_rv=reviews.get(selected_cam)
        top_bar,icon_rects=draw_top_bar(total_w, active_panel, edit_mode, bool(sel_rv and sel_rv["paused"]),
                                        hov if isinstance(hov,str) else None, zones_paused(now_ts))
        layout["icon_rects"]=icon_rects

        playback_bar,playback_rects,timeline=draw_playback_bar(
            total_w, bool(sel_rv and sel_rv["paused"]), cameras[selected_cam]["name"], focus_cam is not None,
            hov[1] if isinstance(hov,tuple) else None, sel_rv,
            readers[selected_cam].spans() if rec_on else [], now_ts, rec_on)
        layout["timeline"]=timeline
        layout["playback_rects"]=playback_rects

        combined=np.vstack([top_bar,body,playback_bar]); total_h=combined.shape[0]

        if zone_menu_visible:
            mx,my=zone_menu_pos; menu_h=MENU_PAD*2+MENU_ITEM_H*len(ZONE_MENU_ITEMS)
            mx=min(max(mx,5),grid_w-MENU_W-5); my=min(max(my,5),total_h-menu_h-5)
            zone_menu_pos=(mx,my); draw_zone_menu(combined,mx,my)

        if isinstance(hov,str) and hov in icon_rects:
            rx1,_,rx2,_ = icon_rects[hov]
            draw_tooltip(combined, TOP_BAR_TIPS.get(hov,""), (rx1+rx2)//2, TOP_BAR_H+6)
        elif isinstance(hov,tuple) and hov[1] in playback_rects:
            rx1,_,rx2,_ = playback_rects[hov[1]]
            if hov[1]=="timeline" and timeline:
                t0,t1,tx1,tx2 = timeline; mx = layout.get("mouse",(0,0))[0]
                tt = t0 + (t1-t0)*min(1.0,max(0.0,(mx-tx1)/max(1,tx2-tx1)))
                tip = "LIVE" if tt >= now_ts-LIVE_EDGE_S else _clock(tt, datetime.fromtimestamp(tt).date()!=datetime.now().date())
                draw_tooltip(combined, tip, mx, total_h-PLAYBACK_BAR_H-30)
            else:
                tip = "Back to grid (Esc)" if (hov[1]=="focus" and focus_cam is not None) else PLAYBACK_TIPS.get(hov[1],"")
                draw_tooltip(combined, tip, (rx1+rx2)//2, total_h-PLAYBACK_BAR_H-34)

        if cfg.get("remote_enabled"): publish_web_frame(grid)

        cv2.imshow(WIN,combined)
        if not _icon_applied:
            set_opencv_window_icon(WIN, ICON_PATH)
            _icon_applied = True

        # Sleep only what's left of this frame's time slot - waiting the full
        # frame_delay on top of the processing time is what capped the old
        # loop at ~17 fps even with a 30 fps source.
        spent_ms=(time.time()-now_ts)*1000
        key=cv2.waitKey(max(1,int(frame_delay-spent_ms)))&0xFF
        if   key==ord('q'): quit_requested=True
        elif key==ord('b'): back_requested=True
        elif key==ord('e'):
            edit_mode=not edit_mode; drawing_zone=False; new_zone_pts=[]
            zone_menu_visible=False; zone_editor.deselect()
            add_log(f"Edit mode {'ON' if edit_mode else 'OFF'}",THEME["accent"])
        elif key==ord('r'):
            cam_tracked.clear(); cam_nid.clear(); add_log("Tracking reset",THEME["warn"])
        elif key==127:
            del_ci = zone_editor.selected_index if zone_editor.selected_index is not None else active_draw_cam
            zone_editor.delete_selected(del_ci)
            editing_zone = None   # indexes shift after a delete
            zone_editor.save_to(cfg); save_config(cfg)
        elif key==ord('+'):
            rz_ci = zone_editor.selected_index if zone_editor.selected_index is not None else active_draw_cam
            zone_editor.resize_selected(rz_ci,1.1); zone_editor.save_to(cfg); save_config(cfg)
        elif key==ord('-'):
            rz_ci = zone_editor.selected_index if zone_editor.selected_index is not None else active_draw_cam
            zone_editor.resize_selected(rz_ci,0.9); zone_editor.save_to(cfg); save_config(cfg)
        elif key==ord(' '):
            review_cmd("pauseplay")
        elif key==ord('l'):
            review_cmd("live")
        elif key==ord('s'):
            take_snapshot()
        elif key==27:
            focus_cam=None
        elif key==ord('f'):
            toggle_focus(selected_cam)
        elif key==82 or key==0:
            log_scroll_pos[0]=min(log_scroll_pos[0]+1,max(0,len(admin_log)-1))
        elif key==84 or key==1:
            log_scroll_pos[0]=max(log_scroll_pos[0]-1,0)
        try:
            if cv2.getWindowProperty(WIN,cv2.WND_PROP_VISIBLE)<1: break
        except cv2.error: break

    person_det.stop()
    for rec in recorders:
        if rec: rec.close()
    for cap in caps:
        if cap: cap.release()
    cv2.destroyAllWindows()
    return back_requested

# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform == "win32":
        # Without this, Windows treats the whole app as DPI-unaware and
        # bitmap-stretches every window to match the display's scale factor
        # (125%/150%/etc, the default on most laptops) — that's what actually
        # causes the blurry/"pixelated" look, not the image resolution itself.
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    try:
        person_model=load_models()
        while True:
            app=SetupApp()
            cfg=app.run()
            if cfg is None:
                break
            if not cfg["cameras"]:
                messagebox.showwarning("Safety Cam","No cameras selected.")
                break
            go_back=run_monitor(cfg,person_model)
            if not go_back:
                break
    except Exception as e:
        import traceback
        with open("error_log.txt","a",encoding="utf-8") as f:
            f.write("\n"+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
            f.write(traceback.format_exc())
        try:    messagebox.showerror("Safety Cam Error",str(e))
        except: print(e)