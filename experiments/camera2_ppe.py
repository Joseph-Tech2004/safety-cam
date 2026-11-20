# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# ── CONFIGURATION ──
CORRIDOR_VID = "Corridor2.mp4"
LOG_PATH = "logs"
os.makedirs(LOG_PATH, exist_ok=True)

person_model = YOLO("yolov8n.pt").to('cuda')
ppe_model    = YOLO("best.pt").to('cuda')

print("Models loaded successfully on GPU hardware pipeline.")

# ── SHARED STATE MEMORY MATRIX ──
system_memory = {
    "has_exited_doorway": False,  
    "ppe_verified_lock": False,
    "last_status": "",
    "last_log_time": 0,
    "admin_logs": [],
    "tracking_active": True,
    "confirmation_counter": 75   # Clear frame countdown
}

MAX_LOG_LINES = 15
DOOR_X_THRESHOLD = 530  
MIN_WIDTH_FOR_CHECK = 75  

def add_admin_log(text, color=(255, 255, 255)):
    now = datetime.now().strftime("%H:%M:%S")
    system_memory["admin_logs"].append({"text": f"[{now}] {text}", "color": color})
    if len(system_memory["admin_logs"]) > MAX_LOG_LINES:
        system_memory["admin_logs"].pop(0)
    print(f"SYSTEM LOG: {text}")

def log_event_file(status):
    now = datetime.now()
    ts = now.timestamp()
    if status != system_memory["last_status"] or (ts - system_memory["last_log_time"]) > 10:
        system_memory["last_status"] = status
        system_memory["last_log_time"] = ts
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        with open(f"{LOG_PATH}/ppe_compliance_{date}.txt", "a") as f:
            f.write(f"{time},{status}\n")

def draw_admin_log_window():
    win_h, win_w = 480, 450
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (win_w, 45), (30, 30, 60), -1)
    cv2.putText(canvas, "SAFETY COMPLIANCE LOG", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.line(canvas, (0, 46), (win_w, 46), (50, 50, 80), 1)
    
    y = 80
    for entry in reversed(system_memory["admin_logs"]):
        cv2.putText(canvas, entry["text"], (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, entry["color"], 1)
        y += 24
        if y > win_h - 20: break
    return canvas

def detect_helmet_ai(frame, x1, y1, x2, y2):
    ph = y2 - y1
    head_region = y1 + int(ph * 0.40)
    crop = frame[y1:head_region, x1:x2]
    if crop.size == 0: return False
    results = ppe_model(crop, conf=0.35, verbose=False)
    for r in results:
        for box in r.boxes:
            if ppe_model.names[int(box.cls[0])] == "Hardhat":
                return True
    return False

def detect_vest_color(frame, x1, y1, x2, y2):
    ph = y2 - y1
    t_top = y1 + int(ph * 0.20)
    t_bot = y1 + int(ph * 0.65)
    torso = frame[t_top:t_bot, x1:x2]
    if torso.size == 0: return False, t_top, t_bot
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([20, 35, 35]), np.array([75, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([5, 120, 120]), np.array([25, 255, 255]))
    m3 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([5, 255, 255]))
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    return cv2.countNonZero(combined_mask) > 500, t_top, t_bot

def main():
    cap = cv2.VideoCapture(CORRIDOR_VID)
    frame_cnt = 0
    has_helmet, has_vest, person_found = False, False, False
    px1, py1, px2, py2 = 0, 0, 0, 0
    t_top, t_bot = 0, 0

    add_admin_log("PPE Monitoring Node Online.", color=(100, 200, 255))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        
        # Core Object Localization loop processing thread
        if system_memory["tracking_active"] and (frame_cnt % 3 == 0):
            p_results = person_model(frame, conf=0.45, verbose=False)
            current_frame_has_human = False
            
            for r in p_results:
                for box in r.boxes:
                    if person_model.names[int(box.cls[0])] == "person":
                        current_frame_has_human = True
                        person_found = True
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        pw = px2 - px1
                        
                        if not system_memory["has_exited_doorway"] and px1 < DOOR_X_THRESHOLD:
                            system_memory["has_exited_doorway"] = True
                            add_admin_log("EVENT: Worker stepping out of changing booth.", color=(0, 165, 255))

                        # CRITICAL BUG FIX: If we already verified compliance once, do not check the width variables again!
                        if system_memory["ppe_verified_lock"]:
                            break

                        if system_memory["has_exited_doorway"] and pw >= MIN_WIDTH_FOR_CHECK:
                            has_helmet = detect_helmet_ai(frame, px1, py1, px2, py2)
                            has_vest, t_top, t_bot = detect_vest_color(frame, px1, py1, px2, py2)
                            
                            if has_helmet and has_vest:
                                system_memory["ppe_verified_lock"] = True
                                add_admin_log("PASSED: Worker is fully PPE Compliant.", color=(0, 255, 0))
                        break
            
            # The system will only reset if the frame becomes empty BEFORE verification is secured
            if not current_frame_has_human and person_found and not system_memory["ppe_verified_lock"]:
                person_found = False
                system_memory["has_exited_doorway"] = False
                has_helmet, has_vest = False, False

        # ── RUNS SMOOTHLY AT NATIVE 30 FPS DURING COUNTDOWN ──
        if system_memory["ppe_verified_lock"] and system_memory["tracking_active"]:
            system_memory["confirmation_counter"] -= 1
            if system_memory["confirmation_counter"] <= 0:
                system_memory["tracking_active"] = False
                person_found = False  # Deletes bounding boxes permanently

        # ── RENDERING OVERLAYS ──
        if person_found and system_memory["tracking_active"]:
            pw = px2 - px1
            if not system_memory["has_exited_doorway"] or (pw < MIN_WIDTH_FOR_CHECK and not system_memory["ppe_verified_lock"]):
                box_color = (0, 165, 255)  
                status_lbl = "SCANNING SAFETY CLOTHING..."
            else:
                box_color = (0, 255, 0) if system_memory["ppe_verified_lock"] else (0, 0, 255)
                status_lbl = "VERIFIED COMPLIANT" if system_memory["ppe_verified_lock"] else "PPE VIOLATION"
                
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, status_lbl, (px1, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            if system_memory["has_exited_doorway"] and (pw >= MIN_WIDTH_FOR_CHECK or system_memory["ppe_verified_lock"]):
                ph = py2 - py1
                head_y2 = py1 + int(ph * 0.40)
                hc = (0, 255, 0) if system_memory["ppe_verified_lock"] else (0, 0, 255)
                cv2.rectangle(frame, (px1, py1), (px2, head_y2), hc, 1)

                vc = (0, 255, 0) if system_memory["ppe_verified_lock"] else (0, 0, 255)
                if t_top > 0 and t_bot > 0:
                    cv2.rectangle(frame, (px1, t_top), (px2, t_bot), vc, 1)

        # ── HUD BANNER CONTROLLER ──
        cv2.rectangle(frame, (0, 0), (640, 42), (20, 20, 20), -1)
        pw = px2 - px1 if person_found else 0
        
        if not system_memory["tracking_active"]:
            status, color = "PPE COMPLIANT - ACCESS TRACK CLOSED", (0, 255, 0)
        elif not person_found:
            status, color = "NO PERSON DETECTED", (150, 150, 150)
        elif not system_memory["has_exited_doorway"] or (pw < MIN_WIDTH_FOR_CHECK and not system_memory["ppe_verified_lock"]):
            status, color = "INITIALIZING SAFETY GEAR MONITORING...", (0, 165, 255)
        elif system_memory["ppe_verified_lock"]:
            status, color = "PPE COMPLIANT - ENTRY ALLOWED", (0, 255, 0)
        elif not has_helmet and not has_vest:
            status, color = "VIOLATION DETECTED: NO HELMET & NO VEST", (0, 0, 255)
        elif not has_helmet:
            status, color = "VIOLATION DETECTED: NO SAFETY HELMET", (0, 0, 255)
        else:
            status, color = "VIOLATION DETECTED: NO SAFETY VEST", (0, 0, 255)

        cv2.putText(frame, f"CAM 3: {status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        if person_found and system_memory["has_exited_doorway"] and system_memory["tracking_active"]:
            log_event_file(status)

        log_canvas = draw_admin_log_window()
        cv2.imshow("Camera 3 - PPE Safety Monitor Window", frame)
        cv2.imshow("System Log Console Dashboard", log_canvas)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): break
        frame_cnt += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()