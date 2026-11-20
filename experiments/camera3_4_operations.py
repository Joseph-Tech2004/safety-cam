# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import time

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
WORKSHOP_VID  = "Corridor8.mp4"
LOG_PATH      = "logs"
os.makedirs(LOG_PATH, exist_ok=True)

# ── Detection zones (640x480) ─────────────────────────────────────────────────
OSC_ZONE = (340, 168, 380, 210)   # oscilloscope screen
BOX_ZONE = (480, 120, 580, 230)   # orange toolbox

# PPE check every 30 seconds
PPE_CHECK_INTERVAL = 30

# ── Models ────────────────────────────────────────────────────────────────────
person_model = YOLO("yolov8n.pt").to('cuda')
ppe_model    = YOLO("best.pt").to('cuda')
print("Models loaded on GPU — Camera 4 Workshop Monitor")

# ── System state ──────────────────────────────────────────────────────────────
state = {
    "osc_on":          False,
    "box_open":        False,
    "session_started": False,
    "session_ended":   False,
    "last_ppe_check":  0,
    "ppe_ok":          True,
    "person_present":  False,
    "alert_sent":      False,
    "admin_logs":      [],
}

MAX_LOG = 18

# ── Admin log ─────────────────────────────────────────────────────────────────
def add_log(text, color=(255,255,255)):
    now = datetime.now().strftime("%H:%M:%S")
    state["admin_logs"].append({"text": f"[{now}] {text}", "color": color})
    if len(state["admin_logs"]) > MAX_LOG:
        state["admin_logs"].pop(0)
    print(f"CAM4 LOG: {text}")

def log_to_file(event, detail=""):
    now  = datetime.now()
    date = now.strftime("%Y-%m-%d")
    ts   = now.strftime("%H:%M:%S")
    with open(f"{LOG_PATH}/workshop_{date}.txt", "a") as f:
        f.write(f"{ts},{event},{detail}\n")

def draw_admin_log():
    win_h, win_w = 520, 480
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    # Title
    cv2.rectangle(canvas, (0,0), (win_w,45), (30,30,60), -1)
    cv2.putText(canvas, "CAMERA 4 - WORKSHOP LOG",
                (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)
    cv2.line(canvas, (0,46), (win_w,46), (50,50,80), 1)

    # Equipment status panel
    cv2.rectangle(canvas, (0,47), (win_w,135), (15,15,35), -1)

    osc_color = (0,255,0)   if state["osc_on"]        else (0,0,255)
    box_color = (0,200,255) if state["box_open"]       else (0,200,100)
    ppe_color = (0,255,0)   if state["ppe_ok"]         else (0,0,255)
    ses_color = (0,255,0)   if state["session_started"] else (150,150,150)

    cv2.putText(canvas,
        f"Oscilloscope: {'ON' if state['osc_on'] else 'OFF'}",
        (8,68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, osc_color, 1)
    cv2.putText(canvas,
        f"Toolbox: {'OPEN' if state['box_open'] else 'CLOSED'}",
        (8,90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
    cv2.putText(canvas,
        f"PPE Status: {'OK' if state['ppe_ok'] else 'VIOLATION'}",
        (8,112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppe_color, 1)
    cv2.putText(canvas,
        f"Session: {'ACTIVE' if state['session_started'] else 'IDLE'}",
        (250,68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ses_color, 1)

    elapsed   = time.time() - state["last_ppe_check"]
    remaining = max(0, PPE_CHECK_INTERVAL - int(elapsed))
    cv2.putText(canvas,
        f"PPE check: {remaining}s",
        (250,90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150,150,200), 1)

    cv2.line(canvas, (0,135), (win_w,135), (50,50,80), 1)

    # Log entries
    y = 158
    for entry in reversed(state["admin_logs"][-14:]):
        text  = entry["text"]
        color = entry["color"]
        if len(text) > 52:
            cv2.putText(canvas, text[:52],
                        (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1)
            y += 16
            cv2.putText(canvas, "   "+text[52:],
                        (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        tuple(int(c*0.7) for c in color), 1)
        else:
            cv2.putText(canvas, text,
                        (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1)
        y += 20
        if y > win_h - 20:
            break

    cv2.putText(canvas, "Q = Quit",
                (8, win_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1)
    return canvas

# ── Oscilloscope detection ────────────────────────────────────────────────────
def detect_oscilloscope_on(frame):
    x1,y1,x2,y2 = OSC_ZONE
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Yellow signal trace
    yellow_mask = cv2.inRange(hsv,
        np.array([18, 80, 80]),
        np.array([45, 255, 255]))

    # Blue screen background
    blue_mask = cv2.inRange(hsv,
        np.array([85, 30, 30]),
        np.array([135, 255, 255]))

    # General screen brightness
    bright_mask  = cv2.inRange(gray, 60, 255)
    total_px     = max((x2-x1)*(y2-y1), 1)
    bright_ratio = cv2.countNonZero(bright_mask) / total_px
    yellow_px    = cv2.countNonZero(yellow_mask)
    blue_ratio   = cv2.countNonZero(blue_mask) / total_px

    return yellow_px > 20 or blue_ratio > 0.10 or bright_ratio > 0.30

# ── Toolbox detection ─────────────────────────────────────────────────────────
def detect_toolbox_open(frame):
    x1,y1,x2,y2 = BOX_ZONE
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Orange color of toolbox lid
    m1 = cv2.inRange(hsv,
        np.array([5,  150, 100]),
        np.array([20, 255, 255]))
    m2 = cv2.inRange(hsv,
        np.array([0,  150, 100]),
        np.array([5,  255, 255]))
    combined = cv2.bitwise_or(m1, m2)

    # Upper 40% = lid area (only visible when open)
    h_roi      = y2 - y1
    upper_h    = int(h_roi * 0.40)
    upper_mask = combined[:upper_h, :]
    lower_mask = combined[upper_h:, :]

    upper_orange = cv2.countNonZero(upper_mask)
    lower_orange = cv2.countNonZero(lower_mask)

    # Open: significant orange in upper zone AND more than lower
    return upper_orange > 200 and upper_orange > lower_orange * 0.3

# ── PPE check ─────────────────────────────────────────────────────────────────
def check_ppe(frame, x1, y1, x2, y2):
    ph = y2 - y1

    # Helmet — AI model
    head_y2    = y1 + int(ph * 0.40)
    crop       = frame[y1:head_y2, x1:x2]
    has_helmet = False
    if crop.size > 0:
        results = ppe_model(crop, conf=0.35, verbose=False)
        for r in results:
            for box in r.boxes:
                if ppe_model.names[int(box.cls[0])] == "Hardhat":
                    has_helmet = True

    # Vest — color detection
    t_top = y1 + int(ph * 0.20)
    t_bot = y1 + int(ph * 0.65)
    torso = frame[t_top:t_bot, x1:x2]
    has_vest = False
    if torso.size > 0:
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, np.array([20,35,35]),  np.array([75,255,255]))
        m2  = cv2.inRange(hsv, np.array([5,120,120]), np.array([25,255,255]))
        m3  = cv2.inRange(hsv, np.array([0,120,120]), np.array([5,255,255]))
        combined = cv2.bitwise_or(cv2.bitwise_or(m1,m2),m3)
        has_vest = cv2.countNonZero(combined) > 400

    return has_helmet, has_vest

# ── End of shift check ────────────────────────────────────────────────────────
def check_end_of_shift():
    violations = []
    if state["osc_on"]:
        violations.append("Oscilloscope still ON")
    if state["box_open"]:
        violations.append("Toolbox still OPEN")
    return violations

# ── Draw zones ────────────────────────────────────────────────────────────────
def draw_zones(frame):
    # Oscilloscope zone
    osc_color = (0,255,0) if state["osc_on"] else (80,80,80)
    cv2.rectangle(frame,
        (OSC_ZONE[0],OSC_ZONE[1]),
        (OSC_ZONE[2],OSC_ZONE[3]),
        osc_color, 1)
    cv2.putText(frame,
        "OSC " + ("ON" if state["osc_on"] else "OFF"),
        (OSC_ZONE[0], OSC_ZONE[1]-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, osc_color, 1)

    # Toolbox zone
    box_color = (0,200,255) if state["box_open"] else (80,80,80)
    cv2.rectangle(frame,
        (BOX_ZONE[0],BOX_ZONE[1]),
        (BOX_ZONE[2],BOX_ZONE[3]),
        box_color, 1)
    cv2.putText(frame,
        "BOX " + ("OPEN" if state["box_open"] else "CLOSED"),
        (BOX_ZONE[0], BOX_ZONE[1]-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1)

    # Lid threshold line — yellow dashed line inside box zone
    bx1,by1,bx2,by2 = BOX_ZONE
    lid_y = by1 + int((by2-by1) * 0.40)
    cv2.line(frame, (bx1, lid_y), (bx2, lid_y), (0,255,255), 1)
    cv2.putText(frame, "LID",
        (bx1, lid_y-3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,255,255), 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(WORKSHOP_VID)
    if not cap.isOpened():
        print(f"Cannot open {WORKSHOP_VID}")
        return

    frame_cnt    = 0
    DETECT_EVERY = 4
    px1,py1,px2,py2 = 0,0,0,0
    blink        = True
    blink_cnt    = 0

    state["last_ppe_check"] = time.time()
    add_log("Workshop monitoring started", color=(100,200,255))
    add_log("Waiting for work session...", color=(150,150,150))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame  = cv2.resize(frame, (640,480))
        now_ts = time.time()

        # ── Equipment detection ───────────────────────────────────────────
        osc_on   = detect_oscilloscope_on(frame)
        box_open = detect_toolbox_open(frame)

        # Oscilloscope state change
        if osc_on != state["osc_on"]:
            state["osc_on"] = osc_on
            if osc_on:
                add_log("Oscilloscope TURNED ON", color=(0,255,200))
                log_to_file("OSC_ON")
                if not state["session_started"]:
                    state["session_started"] = True
                    state["session_ended"]   = False
                    state["alert_sent"]      = False
                    add_log("Work session STARTED", color=(0,255,0))
                    log_to_file("SESSION_START")
            else:
                add_log("Oscilloscope TURNED OFF", color=(0,100,255))
                log_to_file("OSC_OFF")

        # Toolbox state change
        if box_open != state["box_open"]:
            state["box_open"] = box_open
            if box_open:
                add_log("Toolbox OPENED", color=(0,200,255))
                log_to_file("BOX_OPEN")
            else:
                add_log("Toolbox CLOSED", color=(0,255,100))
                log_to_file("BOX_CLOSED")

        # ── Person detection ──────────────────────────────────────────────
        if frame_cnt % DETECT_EVERY == 0:
            p_results         = person_model(frame, conf=0.45, verbose=False)
            person_this_frame = False

            for r in p_results:
                for box in r.boxes:
                    if person_model.names[int(box.cls[0])] == "person":
                        person_this_frame = True
                        px1,py1,px2,py2 = map(int, box.xyxy[0])
                        break

            # Person just left frame
            if not person_this_frame and state["person_present"]:
                state["person_present"] = False
                add_log("Worker left workshop", color=(150,150,150))

                # End of shift check
                if state["session_started"] and not state["session_ended"]:
                    violations = check_end_of_shift()
                    if violations:
                        state["session_ended"] = True
                        msg = "END OF SHIFT VIOLATIONS: " + " | ".join(violations)
                        add_log(msg, color=(0,0,255))
                        log_to_file("END_SHIFT_VIOLATION", ";".join(violations))
                        print(f"\n*** ALERT TO FOREMAN: {msg} ***\n")
                        # Uncomment when Telegram ready:
                        # send_telegram(f"FOREMAN ALERT:\n{msg}")
                    else:
                        state["session_ended"] = True
                        add_log("Shift ended — all equipment secured", color=(0,255,0))
                        log_to_file("END_SHIFT_OK")

            if person_this_frame:
                state["person_present"] = True

                # Periodic PPE check every 1 minute
                if (now_ts - state["last_ppe_check"]) > PPE_CHECK_INTERVAL:
                    state["last_ppe_check"] = now_ts
                    has_helmet, has_vest    = check_ppe(frame, px1,py1,px2,py2)
                    state["ppe_ok"]         = has_helmet and has_vest

                    if state["ppe_ok"]:
                        add_log("PPE periodic check: PASSED", color=(0,255,0))
                        log_to_file("PPE_CHECK_PASS")
                    else:
                        missing = []
                        if not has_helmet: missing.append("helmet")
                        if not has_vest:   missing.append("vest")
                        msg = "PPE VIOLATION: Missing " + " & ".join(missing)
                        add_log(msg, color=(0,0,255))
                        log_to_file("PPE_VIOLATION", ";".join(missing))
                        print(f"\n*** ALERT TO WORKER: {msg} ***\n")
                        # Uncomment when Telegram ready:
                        # send_telegram(f"WORKER ALERT:\n{msg}\nReturn to changing room.")

        # ── Draw zones ────────────────────────────────────────────────────
        draw_zones(frame)

        # ── Draw person box with blink ────────────────────────────────────
        blink_cnt += 1
        if blink_cnt % 15 == 0:
            blink = not blink

        if state["person_present"]:
            p_color = (0,255,0) if state["ppe_ok"] else (0,0,255)
            if blink:
                cv2.rectangle(frame, (px1,py1), (px2,py2), p_color, 2)
                cv2.putText(frame,
                    "PPE OK" if state["ppe_ok"] else "PPE VIOLATION",
                    (px1, py1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 2)
            else:
                cv2.rectangle(frame, (px1,py1), (px2,py2),
                    tuple(c//3 for c in p_color), 1)

        # ── HUD banner ────────────────────────────────────────────────────
        cv2.rectangle(frame, (0,0), (640,42), (20,20,20), -1)

        if not state["session_started"]:
            hud_text  = "CAM4: IDLE - WAITING FOR SESSION"
            hud_color = (150,150,150)
        elif state["session_ended"]:
            viol = check_end_of_shift()
            hud_text  = "CAM4: SHIFT ENDED - " + ("VIOLATIONS!" if viol else "ALL SECURED")
            hud_color = (0,0,255) if viol else (0,255,0)
        elif not state["ppe_ok"]:
            hud_text  = "CAM4: PPE VIOLATION DETECTED"
            hud_color = (0,0,255)
        elif state["osc_on"] and state["box_open"]:
            hud_text  = "CAM4: ACTIVE - OSC ON | BOX OPEN"
            hud_color = (0,255,200)
        elif state["osc_on"]:
            hud_text  = "CAM4: ACTIVE - OSC ON | BOX CLOSED"
            hud_color = (0,200,255)
        else:
            hud_text  = "CAM4: SESSION ACTIVE - MONITORING"
            hud_color = (0,165,255)

        cv2.putText(frame, hud_text,
                    (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)

        # ── Bottom status bar ─────────────────────────────────────────────
        cv2.rectangle(frame, (0,440), (640,480), (15,15,15), -1)

        osc_ind = (0,255,0)   if state["osc_on"]   else (60,60,60)
        box_ind = (0,200,255) if state["box_open"]  else (60,60,60)
        ppe_ind = (0,255,0)   if state["ppe_ok"]    else (0,0,255)

        cv2.circle(frame, (20,460),  8, osc_ind, -1)
        cv2.putText(frame, "OSC", (32,465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, osc_ind, 1)
        cv2.circle(frame, (100,460), 8, box_ind, -1)
        cv2.putText(frame, "BOX", (112,465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_ind, 1)
        cv2.circle(frame, (180,460), 8, ppe_ind, -1)
        cv2.putText(frame, "PPE", (192,465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ppe_ind, 1)

        remaining = max(0, PPE_CHECK_INTERVAL - int(now_ts - state["last_ppe_check"]))
        cv2.putText(frame, f"PPE check: {remaining}s",
                    (420,465), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)

        log_canvas = draw_admin_log()
        cv2.imshow("Camera 4 - Workshop Monitor", frame)
        cv2.imshow("Camera 4 - Admin Log",        log_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_cnt += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()