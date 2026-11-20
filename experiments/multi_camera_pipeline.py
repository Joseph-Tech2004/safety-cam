# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# ── CONFIGURATION ──
CORRIDOR_VID = "Corridor9.mp4"
LOG_PATH = "logs"
os.makedirs(LOG_PATH, exist_ok=True)

person_model = YOLO("yolov8n.pt").to('cuda')
ppe_model    = YOLO("best.pt").to('cuda')

print("Models loaded successfully. Multi-stage continuous compliance logic active.")

# ── SYSTEM MEMORY STATE MATRIX ──
system_memory = {
    "worker_activated": False,    
    "has_exited_doorway": False,  
    "ppe_final_judgement": False, 
    "ppe_status_pass": False,     
    "missing_helmet": False,
    "missing_vest": False,
    "last_status": "",
    "last_log_time": 0,
    "admin_logs": []
}

MAX_LOG_LINES = 15
LOCKER_DOOR_ROI = (455, 175, 530, 395)  
CLOSE_RANGE_X_TRIGGER = 360   
LEFT_EXIT_THRESHOLD = 40  # Threshold broadened to guarantee cleanup on fast motion off-screen

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
    head_region = y1 + int(ph * 0.45)
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
    return cv2.countNonZero(combined_mask) > 400, t_top, t_bot

def main():
    cap = cv2.VideoCapture(CORRIDOR_VID)
    frame_cnt = 0
    person_found = False
    px1, py1, px2, py2 = 0, 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        cv2.rectangle(frame, (LOCKER_DOOR_ROI[0], LOCKER_DOOR_ROI[1]), (LOCKER_DOOR_ROI[2], LOCKER_DOOR_ROI[3]), (255, 255, 0), 1)

        if frame_cnt % 3 == 0:
            p_results = person_model(frame, conf=0.45, verbose=False)
            current_frame_has_human = False
            
            for r in p_results:
                for box in r.boxes:
                    if person_model.names[int(box.cls[0])] == "person":
                        x1_t, y1_t, x2_t, y2_t = map(int, box.xyxy[0])
                        
                        if not system_memory["worker_activated"]:
                            if x2_t >= LOCKER_DOOR_ROI[0] and x1_t <= LOCKER_DOOR_ROI[2]:
                                system_memory["worker_activated"] = True
                                add_admin_log("SYSTEM TRIGGERED: Worker exiting locker room.", color=(255, 150, 0))
                        
                        if not system_memory["worker_activated"]:
                            continue

                        current_frame_has_human = True
                        person_found = True
                        px1, py1, px2, py2 = x1_t, y1_t, x2_t, y2_t
                        
                        if not system_memory["has_exited_doorway"] and px1 < LOCKER_DOOR_ROI[0]:
                            system_memory["has_exited_doorway"] = True
                            
                        break 
            
            # ── GEOMETRIC SCANNING CONDITIONS ──
            if person_found and system_memory["has_exited_doorway"] and not system_memory["ppe_final_judgement"]:
                if px1 < CLOSE_RANGE_X_TRIGGER:
                    has_helmet = detect_helmet_ai(frame, px1, py1, px2, py2)
                    has_vest, _, _ = detect_vest_color(frame, px1, py1, px2, py2)
                    
                    system_memory["ppe_final_judgement"] = True
                    system_memory["missing_helmet"] = not has_helmet
                    system_memory["missing_vest"] = not has_vest
                    
                    if has_helmet and has_vest:
                        system_memory["ppe_status_pass"] = True
                        add_admin_log("COMPLIANCE PASSED: Access granted to workshop.", color=(0, 255, 0))
                    else:
                        system_memory["ppe_status_pass"] = False
                        add_admin_log("COMPLIANCE FAILED: Safety violation logged.", color=(0, 0, 255))

            # ── INDEPENDENT AUTOMATIC CLEANUP SYSTEM ──
            # Clears state variables instantly whenever the workspace becomes empty
            if not current_frame_has_human and person_found:
                person_found = False
                system_memory["worker_activated"] = False
                system_memory["has_exited_doorway"] = False
                system_memory["ppe_final_judgement"] = False
                system_memory["ppe_status_pass"] = False
                system_memory["missing_helmet"] = False
                system_memory["missing_vest"] = False
                add_admin_log("SYSTEM RESET: Room clear. Monitoring line reactivated.", color=(0, 200, 255))

        # ── RENDERING OVERLAYS ──
        if person_found:
            if not system_memory["ppe_final_judgement"]:
                box_color = (0, 165, 255)  
                status_lbl = "SCANNING SAFETY CLOTHING..."
            else:
                if system_memory["ppe_status_pass"]:
                    box_color = (0, 255, 0)
                    status_lbl = "VERIFIED COMPLIANT"
                else:
                    box_color = (0, 0, 255)
                    if system_memory["missing_helmet"] and system_memory["missing_vest"]:
                        status_lbl = "ALARM: NO HELMET & NO VEST"
                    elif system_memory["missing_helmet"]:
                        status_lbl = "ALARM: MISSING HELMET"
                    else:
                        status_lbl = "ALARM: MISSING SAFETY VEST"
                
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, status_lbl, (px1, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # ── HUD BANNER CONTROLLER ──
        cv2.rectangle(frame, (0, 0), (640, 42), (20, 20, 20), -1)
        if not person_found:
            status, color = "SYSTEM IDLE - AWAITING WORKER DEPARTURE", (150, 150, 150)
        elif not system_memory["ppe_final_judgement"]:
            status, color = "PROXIMITY SCANNING IN PROGRESS...", (0, 165, 255)
        elif system_memory["ppe_status_pass"]:
            status, color = "PPE COMPLIANT - PASS VERIFIED", (0, 255, 0)
        else:
            if system_memory["missing_helmet"] and system_memory["missing_vest"]:
                status, color = "VIOLATION DETECTED: NO HELMET & NO VEST", (0, 0, 255)
            elif system_memory["missing_helmet"]:
                status, color = "VIOLATION DETECTED: NO SAFETY HELMET", (0, 0, 255)
            else:
                status, color = "VIOLATION DETECTED: NO SAFETY VEST", (0, 0, 255)

        cv2.putText(frame, f"CAM 3: {status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        if person_found and system_memory["has_exited_doorway"]:
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