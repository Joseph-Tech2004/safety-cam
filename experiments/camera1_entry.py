# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
import os, json, time
from datetime import datetime
from collections import defaultdict
import requests
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG_FILE   = "system_config.json"
LOG_PATH      = "logs"
os.makedirs(LOG_PATH, exist_ok=True)

PPE_MODEL_PATH      = r"C:\Users\josef\Downloads\factory-ppe.v1i.yolov8\runs\detect\train-2\weights\best.pt"
WORKSHOP_MODEL_PATH = r"C:\Users\josef\Desktop\Factory_Safety_System\runs\detect\train-6\weights\best.pt"

TELEGRAM_TOKEN   = ""
TELEGRAM_CHAT_ID = ""

GRACE_PERIOD = 10
PPE_INTERVAL = 1.0

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
last_telegram_time = {}

def send_telegram(msg, cooldown=60):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM] {msg}"); return
    key = msg[:30]
    now = time.time()
    if now - last_telegram_time.get(key,0) < cooldown: return
    last_telegram_time[key] = now
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id":TELEGRAM_CHAT_ID,"text":msg}, timeout=5)
        print(f"[TELEGRAM] {msg}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

# ── MODELS ────────────────────────────────────────────────────────────────────
ppe_model      = None
workshop_model = None

def load_models():
    global ppe_model, workshop_model
    person_mdl = YOLO("yolov8n.pt").to('cuda')
    print("Person model loaded")
    if os.path.exists(PPE_MODEL_PATH):
        ppe_model = YOLO(PPE_MODEL_PATH).to('cuda')
        print(f"PPE model: {ppe_model.names}")
    if os.path.exists(WORKSHOP_MODEL_PATH):
        workshop_model = YOLO(WORKSHOP_MODEL_PATH).to('cuda')
        print(f"Workshop model: {workshop_model.names}")
    return person_mdl

# ── ZONE TYPES ────────────────────────────────────────────────────────────────
ZONE_TYPES = {
    "ppe_check":    {"label":"PPE Check",      "color":(0,255,255),  "key":"1"},
    "ppe_trigger":  {"label":"PPE Trigger",    "color":(0,165,255),  "key":"2"},
    "rest_area":    {"label":"Rest Area",      "color":(0,200,100),  "key":"3"},
    "osc_zone":     {"label":"Oscilloscope",   "color":(255,200,0),  "key":"4"},
    "toolbox_zone": {"label":"Toolbox",        "color":(255,100,0),  "key":"5"},
}

# ── CONFIG ────────────────────────────────────────────────────────────────────
def default_config():
    return {"cameras":[],"zones":{},"schedules":[],
            "telegram_token":"","telegram_chat_id":""}

def save_config(cfg):
    with open(CONFIG_FILE,"w") as f: json.dump(cfg,f,indent=2)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE,"r") as f: return json.load(f)
        except: pass
    return default_config()

# ── ZONE EDITOR ───────────────────────────────────────────────────────────────
class ZoneEditor:
    def __init__(self):
        self.zones         = {}
        self.selected_zone = None
        self.selected_pt   = None
        self.DRAG_R        = 10

    def load(self,cfg):
        self.zones={}
        for k,v in cfg.get("zones",{}).items():
            self.zones[int(k)]=v

    def save_to(self,cfg):
        cfg["zones"]={str(k):v for k,v in self.zones.items()}

    def get_zones(self,ci):
        return self.zones.get(ci,[])

    def add_zone(self,ci,ztype,pts,name=""):
        self.zones.setdefault(ci,[]).append({
            "type":ztype,
            "name":name or ZONE_TYPES[ztype]["label"],
            "points":pts,"active":True})

    def delete_zone(self,ci,zi):
        if ci in self.zones and zi<len(self.zones[ci]):
            self.zones[ci].pop(zi)

    def pt_in_zone(self,zone,x,y):
        pts=np.array(zone["points"],dtype=np.int32)
        if len(pts)<3: return False
        return cv2.pointPolygonTest(pts,(float(x),float(y)),False)>=0

    def draw_zones(self,frame,ci,edit_mode=False):
        for zone in self.get_zones(ci):
            pts=np.array(zone["points"],dtype=np.int32)
            col=ZONE_TYPES.get(zone["type"],{}).get("color",(200,200,200))
            if len(pts)>=3:
                ov=frame.copy()
                cv2.fillPoly(ov,[pts],col)
                cv2.addWeighted(ov,0.12,frame,0.88,0,frame)
                cv2.polylines(frame,[pts],True,col,2)
            if len(pts)>0:
                cv2.putText(frame,zone["name"],
                    (pts[0][0],pts[0][1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
            if edit_mode:
                for p in zone["points"]:
                    cv2.circle(frame,tuple(p),7,(0,0,255),-1)
                    cv2.circle(frame,tuple(p),7,(255,255,255),1)

    def seg_dist(self,p,a,b):
        p,a,b=np.array(p,float),np.array(a,float),np.array(b,float)
        ab=b-a; t=np.dot(p-a,ab)/(np.dot(ab,ab)+1e-6)
        return np.linalg.norm(p-(a+max(0,min(1,t))*ab))

    def handle_mouse(self,event,x,y,ci):
        zones=self.get_zones(ci)
        DR=self.DRAG_R
        if event==cv2.EVENT_LBUTTONDOWN:
            for zi,zone in enumerate(zones):
                for pi,p in enumerate(zone["points"]):
                    if abs(p[0]-x)<DR and abs(p[1]-y)<DR:
                        self.selected_zone=zi; self.selected_pt=pi; return
            for zi,zone in enumerate(zones):
                n=len(zone["points"])
                for i in range(n):
                    if self.seg_dist((x,y),zone["points"][i],
                                     zone["points"][(i+1)%n])<8:
                        zone["points"].insert(i+1,[x,y])
                        self.selected_zone=zi; self.selected_pt=i+1; return
        elif event==cv2.EVENT_MOUSEMOVE:
            if self.selected_zone is not None and self.selected_pt is not None:
                zones[self.selected_zone]["points"][self.selected_pt]=[x,y]
        elif event==cv2.EVENT_LBUTTONUP:
            self.selected_zone=None; self.selected_pt=None
        elif event==cv2.EVENT_RBUTTONDOWN:
            for zi,zone in enumerate(zones):
                for pi,p in enumerate(zone["points"]):
                    if abs(p[0]-x)<DR and abs(p[1]-y)<DR and len(zone["points"])>3:
                        zone["points"].pop(pi); return

# ── PPE ───────────────────────────────────────────────────────────────────────
def check_ppe(frame,x1,y1,x2,y2):
    if ppe_model is None: return False,False
    H,W=frame.shape[:2]
    ph=y2-y1
    pad_top=max(60,int(ph*0.25)); pad_side=max(20,int((x2-x1)*0.10))
    ex1=max(0,x1-pad_side); ex2=min(W,x2+pad_side)
    ey1=max(0,y1-pad_top)
    hb=y1+int(ph*0.45)
    hcrop=frame[max(0,ey1):hb,ex1:ex2]
    has_helmet=False
    if hcrop.size>0:
        for r in ppe_model(hcrop,conf=0.35,verbose=False):
            for box in r.boxes:
                if ppe_model.names[int(box.cls[0])]=="Hardhat":
                    has_helmet=True
    torso=frame[y1+int(ph*0.20):y1+int(ph*0.65),ex1:ex2]
    has_vest=False
    if torso.size>0:
        hsv=cv2.cvtColor(torso,cv2.COLOR_BGR2HSV)
        m1=cv2.inRange(hsv,np.array([20,35,35]),np.array([75,255,255]))
        m2=cv2.inRange(hsv,np.array([5,120,120]),np.array([25,255,255]))
        m3=cv2.inRange(hsv,np.array([0,120,120]),np.array([5,255,255]))
        has_vest=cv2.countNonZero(cv2.bitwise_or(cv2.bitwise_or(m1,m2),m3))>400
    return has_helmet,has_vest

def miss(h,v):
    m=[]
    if not h: m.append("HELMET")
    if not v: m.append("VEST")
    return " & ".join(m) if m else ""

# ── WORKSHOP ──────────────────────────────────────────────────────────────────
def check_workshop(frame,zone):
    if workshop_model is None: return None,None
    pts=zone["points"]
    if len(pts)<2: return None,None
    x1=min(p[0] for p in pts); y1=min(p[1] for p in pts)
    x2=max(p[0] for p in pts); y2=max(p[1] for p in pts)
    crop=frame[max(0,y1):y2,max(0,x1):x2]
    if crop.size==0: return None,None
    osc=None; tb=None
    for r in workshop_model(crop,conf=0.40,verbose=False):
        for box in r.boxes:
            cls=workshop_model.names[int(box.cls[0])]
            if "oscilloscope_on"  in cls: osc=True
            if "oscilloscope_off" in cls: osc=False
            if "toolbox_open"     in cls: tb=True
            if "toolbox_closed"   in cls: tb=False
    return osc,tb

# ── SCHEDULES ─────────────────────────────────────────────────────────────────
last_sched_alert={}

def check_schedules(schedules,osc_states,tb_states):
    now=datetime.now()
    for s in schedules:
        key=f"{s['hour']}:{s['minute']:02d}"
        if now.hour==s["hour"] and now.minute==s["minute"]:
            today=now.strftime("%Y-%m-%d")
            if last_sched_alert.get(key)!=today:
                last_sched_alert[key]=today
                msgs=[]
                if s.get("check_osc") and any(osc_states.values()):
                    msgs.append("⚠️ Oscilloscope still ON — please turn it off!")
                if s.get("check_toolbox") and any(tb_states.values()):
                    msgs.append("⚠️ Toolbox still OPEN — please close it!")
                if msgs:
                    msg=f"🏭 Factory Alert [{s.get('label',key)}]:\n"+"\n".join(msgs)
                    send_telegram(msg); add_log(msg,(0,0,255))

# ── TRACKING ──────────────────────────────────────────────────────────────────
cam_tracked = defaultdict(dict)
cam_nid     = defaultdict(lambda:[0])

def get_track(ci,cx,cy):
    for tid,d in cam_tracked[ci].items():
        if abs(d["cx"]-cx)<80 and abs(d["cy"]-cy)<80: return tid
    tid=cam_nid[ci][0]; cam_nid[ci][0]+=1
    cam_tracked[ci][tid]={
        "cx":cx,"cy":cy,"ppe_ok":True,"ppe_confirmed":False,
        "missing":"","violation_start":None,"grace_active":False,
        "last_ppe":0,"cached_h":False,"cached_v":False,"confirm_time":0}
    return tid

# ── ADMIN LOG ─────────────────────────────────────────────────────────────────
admin_log=[]
def add_log(text,color=(255,255,255)):
    now=datetime.now().strftime("%H:%M:%S")
    admin_log.append({"text":f"[{now}] {text}","color":color})
    if len(admin_log)>40: admin_log.pop(0)
    print("LOG:",text)

def log_file(event,detail=""):
    now=datetime.now()
    with open(f"{LOG_PATH}/system_{now.strftime('%Y-%m-%d')}.txt","a") as f:
        f.write(f"{now.strftime('%H:%M:%S')},{event},{detail}\n")

# ── DRAW BOTTOM BAR (log + toolbar) ──────────────────────────────────────────
def draw_bottom_bar(total_w, bar_h, edit_mode, drawing_zone,
                    selected_zone_type, cam_count):
    """
    Left side (70%): admin log
    Right side (30%): zone toolbar
    """
    canvas = np.zeros((bar_h, total_w, 3), dtype=np.uint8)
    log_w  = int(total_w * 0.70)
    tool_w = total_w - log_w

    # ── Admin log panel ───────────────────────────────────────────────────
    cv2.rectangle(canvas,(0,0),(log_w,bar_h),(20,20,40),-1)
    cv2.rectangle(canvas,(0,0),(log_w,28),(30,30,60),-1)
    cv2.putText(canvas,"SYSTEM LOG",(8,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
    cv2.putText(canvas,datetime.now().strftime("%H:%M:%S"),
                (log_w-65,20),cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1)
    y=46
    line_h=18
    max_lines=(bar_h-36)//line_h
    for e in reversed(admin_log[-max_lines:]):
        cv2.putText(canvas,e["text"],(6,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,e["color"],1)
        y+=line_h

    # Divider
    cv2.line(canvas,(log_w,0),(log_w,bar_h),(60,60,80),1)

    # ── Zone toolbar panel ────────────────────────────────────────────────
    tx = log_w
    cv2.rectangle(canvas,(tx,0),(total_w,bar_h),(15,15,35),-1)
    cv2.rectangle(canvas,(tx,0),(total_w,28),(25,25,55),-1)
    cv2.putText(canvas,"ZONE TOOLBAR",(tx+8,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),1)

    # Mode indicator
    if edit_mode:
        cv2.putText(canvas,"[EDIT ON]",(tx+8,44),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1)
    if drawing_zone:
        cv2.putText(canvas,"DRAWING — right-click finish",(tx+8,44),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,100),1)

    # Zone type buttons
    btn_y = 54
    btn_h = 26
    btn_margin = 4
    for ztype, zinfo in ZONE_TYPES.items():
        col   = zinfo["color"]
        label = f"[{zinfo['key']}] {zinfo['label']}"
        bx1   = tx + 6
        bx2   = total_w - 6
        by1   = btn_y
        by2   = btn_y + btn_h
        # Highlight selected
        fill  = col if selected_zone_type==ztype else (30,30,50)
        cv2.rectangle(canvas,(bx1,by1),(bx2,by2),fill,-1)
        cv2.rectangle(canvas,(bx1,by1),(bx2,by2),col,1)
        tc = (0,0,0) if selected_zone_type==ztype else col
        cv2.putText(canvas,label,(bx1+6,by1+17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,tc,1)
        btn_y += btn_h + btn_margin

    # Controls hint
    hint_y = btn_y + 8
    hints = ["E = toggle edit","A = add zone","D = delete zone","R = reset tracks","Q = quit"]
    for h in hints:
        if hint_y > bar_h-14: break
        cv2.putText(canvas,h,(tx+6,hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,80,100),1)
        hint_y += 14

    return canvas, log_w  # return log_w so mouse handler knows boundary

# ── PROCESS FRAME ─────────────────────────────────────────────────────────────
def process_cam_frame(frame, ci, zone_editor, now_ts, person_model):
    zones   = zone_editor.get_zones(ci)
    tracked = cam_tracked[ci]

    if not hasattr(process_cam_frame,'boxes'):
        process_cam_frame.boxes=defaultdict(list)
        process_cam_frame.cnt=defaultdict(int)

    process_cam_frame.cnt[ci]+=1
    if process_cam_frame.cnt[ci] % 3 == 0:
        process_cam_frame.boxes[ci]=[]
        for r in person_model(frame,conf=0.45,verbose=False):
            for box in r.boxes:
                if person_model.names[int(box.cls[0])]=="person":
                    process_cam_frame.boxes[ci].append(tuple(map(int,box.xyxy[0])))

    boxes   = process_cam_frame.boxes.get(ci,[])
    seen_ids= set()
    osc_st  = {}
    tb_st   = {}

    for (x1,y1,x2,y2) in boxes:
        cx,cy=(x1+x2)//2,(y1+y2)//2
        tid=get_track(ci,cx,cy)
        t=tracked[tid]; t["cx"],t["cy"]=cx,cy
        seen_ids.add(tid)

        in_ppe=in_rest=in_trigger=False
        for zone in zones:
            if not zone_editor.pt_in_zone(zone,cx,cy): continue
            zt=zone["type"]
            if zt=="ppe_check":   in_ppe=True
            if zt=="rest_area":   in_rest=True
            if zt=="ppe_trigger": in_trigger=True

        if in_rest:
            t["grace_active"]=False; t["violation_start"]=None
            t["ppe_confirmed"]=False
            bc=(0,200,100); lbl="REST AREA"
        elif in_ppe or in_trigger:
            h=t["cached_h"]; v=t["cached_v"]
            if now_ts-t["last_ppe"]>PPE_INTERVAL:
                h,v=check_ppe(frame,x1,y1,x2,y2)
                t["cached_h"]=h; t["cached_v"]=v; t["last_ppe"]=now_ts
            ok=h and v
            if ok:
                t["ppe_ok"]=True; t["ppe_confirmed"]=True
                t["missing"]=""; t["violation_start"]=None
                t["grace_active"]=False; t["confirm_time"]=now_ts
                bc=(0,255,0); lbl="PPE OK"
            else:
                if t["ppe_confirmed"]:
                    if t["violation_start"] is None:
                        t["violation_start"]=now_ts; t["grace_active"]=True
                    elapsed=now_ts-t["violation_start"]
                    if elapsed<GRACE_PERIOD:
                        bc=(0,200,255); lbl=f"CHECKING {GRACE_PERIOD-elapsed:.0f}s"
                    else:
                        t["ppe_ok"]=False; t["missing"]=miss(h,v)
                        t["grace_active"]=False; t["violation_start"]=None
                        t["ppe_confirmed"]=False
                        bc=(0,0,255); lbl=f"VIOLATION: {t['missing']}"
                        add_log(f"CAM{ci+1} VIOLATION: {t['missing']}",(0,0,255))
                        log_file("PPE_VIOLATION",f"cam{ci+1},{t['missing']}")
                        send_telegram(f"⚠️ PPE Violation CAM{ci+1}!\nMissing: {t['missing']}")
                else:
                    t["ppe_ok"]=False; t["missing"]=miss(h,v)
                    bc=(0,0,255); lbl=f"MISSING: {t['missing']}"
        else:
            bc=(0,165,255); lbl="IN FRAME"

        ph=y2-y1; pad_top=max(60,int(ph*0.25))
        dy1=max(0,y1-pad_top)
        cv2.rectangle(frame,(x1,dy1),(x2,y2),bc,2)
        cv2.putText(frame,lbl,(x1,dy1-6),cv2.FONT_HERSHEY_SIMPLEX,0.46,bc,2)

    for tid in list(tracked.keys()):
        if tid not in seen_ids: del tracked[tid]

    # Workshop zones
    for zi,zone in enumerate(zones):
        if zone["type"] in ("osc_zone","toolbox_zone"):
            osc,tb=check_workshop(frame,zone)
            col=ZONE_TYPES[zone["type"]]["color"]
            if zone["type"]=="osc_zone" and osc is not None:
                osc_st[f"{ci}_{zi}"]=osc
                sl="OSC ON" if osc else "OSC OFF"
                sc=(0,100,255) if osc else (0,255,0)
                pts=zone["points"]
                cv2.putText(frame,sl,(pts[0][0],pts[0][1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,sc,2)
            if zone["type"]=="toolbox_zone" and tb is not None:
                tb_st[f"{ci}_{zi}"]=tb
                sl="BOX OPEN" if tb else "BOX CLOSED"
                sc=(0,200,255) if tb else (0,255,100)
                pts=zone["points"]
                cv2.putText(frame,sl,(pts[0][0],pts[0][1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,sc,2)

    return frame, osc_st, tb_st

# ── SETUP GUI ─────────────────────────────────────────────────────────────────
class SetupApp:
    def __init__(self):
        self.cfg=load_config()
        self.root=tk.Tk()
        self.root.title("Factory Safety System - Setup")
        self.root.geometry("700x580")
        self.root.configure(bg="#1a1a2e")
        self._build()

    def _lbl(self,p,t,**kw):
        return tk.Label(p,text=t,bg="#1a1a2e",fg="white",**kw)

    def _btn(self,p,t,cmd,c="#0f3460"):
        return tk.Button(p,text=t,command=cmd,bg=c,fg="white",
                         relief="flat",padx=8,pady=4)

    def _build(self):
        tk.Label(self.root,text="🏭 FACTORY SAFETY MONITOR",
                 bg="#1a1a2e",fg="#00c8ff",
                 font=("Arial",15,"bold")).pack(pady=10)

        # Cameras
        cf=tk.LabelFrame(self.root,text="Cameras (max 4)",
                         bg="#1a1a2e",fg="#00c8ff")
        cf.pack(fill="x",padx=12,pady=4)
        self.cam_lb=tk.Listbox(cf,bg="#0f3460",fg="white",
                               height=4,selectbackground="#e94560")
        self.cam_lb.pack(side="left",fill="both",expand=True,padx=4,pady=4)
        self._refresh_cams()
        bf=tk.Frame(cf,bg="#1a1a2e"); bf.pack(side="left",padx=4)
        self._btn(bf,"+ Live Camera",self._add_live).pack(fill="x",pady=2)
        self._btn(bf,"+ Video File", self._add_video).pack(fill="x",pady=2)
        self._btn(bf,"✕ Remove",     self._rem_cam,"#e94560").pack(fill="x",pady=2)

        # Telegram
        tf=tk.LabelFrame(self.root,text="Telegram",bg="#1a1a2e",fg="#00c8ff")
        tf.pack(fill="x",padx=12,pady=4)
        self._lbl(tf,"Token:").grid(row=0,column=0,sticky="w",padx=4)
        self.tok=tk.StringVar(value=self.cfg.get("telegram_token",""))
        tk.Entry(tf,textvariable=self.tok,width=50,bg="#0f3460",fg="white",
                 insertbackground="white").grid(row=0,column=1,padx=4,pady=2)
        self._lbl(tf,"Chat ID:").grid(row=1,column=0,sticky="w",padx=4)
        self.cid=tk.StringVar(value=self.cfg.get("telegram_chat_id",""))
        tk.Entry(tf,textvariable=self.cid,width=30,bg="#0f3460",fg="white",
                 insertbackground="white").grid(row=1,column=1,padx=4,pady=2,sticky="w")

        # Schedules
        sf=tk.LabelFrame(self.root,text="Schedules",bg="#1a1a2e",fg="#00c8ff")
        sf.pack(fill="x",padx=12,pady=4)
        self.sched_lb=tk.Listbox(sf,bg="#0f3460",fg="white",
                                 height=3,selectbackground="#e94560")
        self.sched_lb.pack(side="left",fill="both",expand=True,padx=4,pady=4)
        self._refresh_scheds()
        sbf=tk.Frame(sf,bg="#1a1a2e"); sbf.pack(side="left",padx=4)
        self._btn(sbf,"+ Add",   self._add_sched).pack(fill="x",pady=2)
        self._btn(sbf,"✕ Remove",self._rem_sched,"#e94560").pack(fill="x",pady=2)

        self._btn(self.root,"▶  LAUNCH",self._launch,"#00b894").pack(pady=12)

    def _refresh_cams(self):
        self.cam_lb.delete(0,"end")
        for i,c in enumerate(self.cfg["cameras"]):
            self.cam_lb.insert("end",f"CAM{i+1}: {c['name']} [{c['source']}]")

    def _refresh_scheds(self):
        self.sched_lb.delete(0,"end")
        for s in self.cfg.get("schedules",[]):
            flags=""
            if s.get("check_osc"):     flags+=" OSC"
            if s.get("check_toolbox"): flags+=" BOX"
            self.sched_lb.insert("end",
                f"{s['hour']:02d}:{s['minute']:02d}  {s.get('label','')}  [{flags.strip()}]")

    def _add_live(self):
        if len(self.cfg["cameras"])>=4:
            messagebox.showwarning("Limit","Max 4 cameras"); return
        idx=simpledialog.askinteger("Index","Camera index:",minvalue=0,maxvalue=9)
        if idx is None: return
        name=simpledialog.askstring("Name","Name:",
             initialvalue=f"CAM{len(self.cfg['cameras'])+1}")
        self.cfg["cameras"].append({"source":str(idx),"name":name or f"CAM{len(self.cfg['cameras'])+1}"})
        save_config(self.cfg); self._refresh_cams()

    def _add_video(self):
        if len(self.cfg["cameras"])>=4:
            messagebox.showwarning("Limit","Max 4 cameras"); return
        path=filedialog.askopenfilename(
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not path: return
        name=simpledialog.askstring("Name","Name:",
             initialvalue=os.path.basename(path).split(".")[0])
        self.cfg["cameras"].append({"source":path,"name":name or "CAM"})
        save_config(self.cfg); self._refresh_cams()

    def _rem_cam(self):
        sel=self.cam_lb.curselection()
        if not sel: return
        self.cfg["cameras"].pop(sel[0])
        save_config(self.cfg); self._refresh_cams()

    def _add_sched(self):
        h=simpledialog.askinteger("Hour","Hour (0-23):",minvalue=0,maxvalue=23)
        if h is None: return
        m=simpledialog.askinteger("Minute","Minute (0-59):",minvalue=0,maxvalue=59)
        if m is None: return
        lbl=simpledialog.askstring("Label","Label:",initialvalue="End of shift")
        co=messagebox.askyesno("OSC","Check oscilloscope?")
        ct=messagebox.askyesno("BOX","Check toolbox?")
        self.cfg.setdefault("schedules",[]).append({
            "hour":h,"minute":m or 0,"label":lbl or "",
            "check_osc":co,"check_toolbox":ct})
        save_config(self.cfg); self._refresh_scheds()

    def _rem_sched(self):
        sel=self.sched_lb.curselection()
        if not sel: return
        self.cfg["schedules"].pop(sel[0])
        save_config(self.cfg); self._refresh_scheds()

    def _launch(self):
        self.cfg["telegram_token"]  =self.tok.get().strip()
        self.cfg["telegram_chat_id"]=self.cid.get().strip()
        save_config(self.cfg); self.root.destroy()

    def run(self):
        self.root.mainloop(); return self.cfg

# ── MAIN MONITOR ──────────────────────────────────────────────────────────────
def run_monitor(cfg, person_model):
    global TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_TOKEN   = cfg.get("telegram_token","")
    TELEGRAM_CHAT_ID = cfg.get("telegram_chat_id","")

    cameras=cfg["cameras"]
    if not cameras: print("No cameras."); return

    caps=[]
    for cam in cameras:
        src=cam["source"]
        try:
            cap=cv2.VideoCapture(int(src),cv2.CAP_MSMF)
        except ValueError:
            cap=cv2.VideoCapture(src)
        caps.append(cap if cap.isOpened() else None)

    fps=30
    for cap in caps:
        if cap:
            f=cap.get(cv2.CAP_PROP_FPS)
            if f>0: fps=f; break
    frame_delay=max(1,int(1000/fps))

    zone_editor=ZoneEditor()
    zone_editor.load(cfg)

    WIN="Factory Safety Monitor"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)

    n=len(cameras)
    cols=2 if n>1 else 1
    rows=(n+cols-1)//cols
    PW,PH=640,480
    BAR_H=220   # bottom bar height

    edit_mode         = False
    drawing_zone      = False
    selected_zone_type= None
    new_zone_pts      = []
    active_draw_cam   = 0
    all_osc           = {}
    all_tb            = {}

    # Mouse state needs panel layout info — store in mutable container
    layout = {"cols":cols,"rows":rows,"PW":PW,"PH":PH,"BAR_H":BAR_H,
              "log_w":0,"total_w":cols*PW}

    def mouse_cb(event,x,y,flags,param):
        nonlocal drawing_zone,new_zone_pts,active_draw_cam,selected_zone_type
        total_w = layout["total_w"]
        bar_top = rows*PH

        # Click in bottom bar — zone type selection
        if y >= bar_top:
            if event==cv2.EVENT_LBUTTONDOWN:
                log_w  = layout["log_w"]
                tx     = log_w
                btn_y  = bar_top+54
                btn_h  = 26
                btn_m  = 4
                for ztype in ZONE_TYPES:
                    bx1=tx+6; bx2=total_w-6
                    by1=btn_y; by2=btn_y+btn_h
                    if bx1<=x<=bx2 and by1<=y<=by2:
                        selected_zone_type=ztype
                        drawing_zone=True
                        new_zone_pts=[]
                        add_log(f"Drawing {ZONE_TYPES[ztype]['label']} — left-click=add, right-click=finish",(0,255,200))
                        return
                    btn_y+=btn_h+btn_m
            return

        # Click in camera panels
        cam_col=x//PW; cam_row=y//PH
        ci=cam_row*cols+cam_col
        if ci>=n: return
        cx=x-cam_col*PW; cy=y-cam_row*PH

        if drawing_zone:
            if event==cv2.EVENT_LBUTTONDOWN:
                active_draw_cam=ci
                new_zone_pts.append([cx,cy])
            elif event==cv2.EVENT_RBUTTONDOWN:
                if len(new_zone_pts)>=3:
                    zone_editor.add_zone(active_draw_cam,selected_zone_type,
                                         new_zone_pts.copy())
                    zone_editor.save_to(cfg); save_config(cfg)
                    add_log(f"Zone saved on CAM{active_draw_cam+1}",(0,200,255))
                drawing_zone=False; new_zone_pts=[]; selected_zone_type=None
        elif edit_mode:
            zone_editor.handle_mouse(event,cx,cy,ci)
            if event==cv2.EVENT_LBUTTONUP:
                zone_editor.save_to(cfg); save_config(cfg)

    cv2.setMouseCallback(WIN,mouse_cb)

    add_log("System started",(100,200,255))
    add_log("Click zone type in toolbar then draw on camera",(150,150,150))
    print("Running — Q=quit  E=edit  R=reset")

    while True:
        frames=[]
        now_ts=time.time()
        all_osc.clear(); all_tb.clear()

        for ci,(cap,cam) in enumerate(zip(caps,cameras)):
            if cap is None:
                blank=np.zeros((PH,PW,3),dtype=np.uint8)
                cv2.putText(blank,"NO SIGNAL",(PW//2-70,PH//2),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(blank,cam["name"],(8,26),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
                frames.append(blank); continue

            ret,frame=cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES,0); ret,frame=cap.read()
            if not ret:
                frames.append(np.zeros((PH,PW,3),dtype=np.uint8)); continue

            frame=cv2.resize(frame,(PW,PH))
            frame,osc_st,tb_st=process_cam_frame(
                frame,ci,zone_editor,now_ts,person_model)
            all_osc.update(osc_st); all_tb.update(tb_st)

            zone_editor.draw_zones(frame,ci,edit_mode)

            # Draw zone being created
            if drawing_zone and active_draw_cam==ci and len(new_zone_pts)>0:
                col=ZONE_TYPES.get(selected_zone_type,{}).get("color",(0,255,0))
                for pt in new_zone_pts:
                    cv2.circle(frame,tuple(pt),5,col,-1)
                if len(new_zone_pts)>1:
                    cv2.polylines(frame,
                        [np.array(new_zone_pts,dtype=np.int32)],False,col,1)

            cv2.rectangle(frame,(0,0),(PW,28),(20,20,20),-1)
            cv2.putText(frame,cam["name"],(8,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),2)
            if edit_mode:
                cv2.putText(frame,"[EDIT]",(PW-60,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
            frames.append(frame)

        check_schedules(cfg.get("schedules",[]),all_osc,all_tb)

        # Fill empty slots
        while len(frames)<cols*rows:
            blank=np.zeros((PH,PW,3),dtype=np.uint8)
            cv2.putText(blank,"NO CAMERA",(PW//2-80,PH//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(40,40,40),2)
            frames.append(blank)

        rows_list=[]
        for r in range(rows):
            row_fs=frames[r*cols:(r+1)*cols]
            while len(row_fs)<cols:
                row_fs.append(np.zeros((PH,PW,3),dtype=np.uint8))
            rows_list.append(np.hstack(row_fs))
        grid=np.vstack(rows_list)

        total_w=grid.shape[1]
        layout["total_w"]=total_w

        bar,log_w=draw_bottom_bar(
            total_w,BAR_H,edit_mode,drawing_zone,selected_zone_type,n)
        layout["log_w"]=log_w

        combined=np.vstack([grid,bar])
        cv2.imshow(WIN,combined)

        key=cv2.waitKey(frame_delay)&0xFF
        if   key==ord('q'): break
        elif key==ord('e'):
            edit_mode=not edit_mode
            drawing_zone=False; new_zone_pts=[]
            add_log(f"Edit mode {'ON — drag red dots' if edit_mode else 'OFF'}",(0,255,255))
        elif key==ord('r'):
            cam_tracked.clear(); cam_nid.clear()
            add_log("Tracking reset",(0,100,255))
        elif key==ord('d'):
            # Delete mode — click zone to delete when edit is on
            add_log("Delete: turn on edit mode (E), then press D + click zone point to remove zone",(200,100,0))

    for cap in caps:
        if cap: cap.release()
    cv2.destroyAllWindows()

# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    person_model = load_models()
    app  = SetupApp()
    cfg  = app.run()
    if cfg["cameras"]:
        run_monitor(cfg, person_model)
    else:
        print("No cameras — exiting.")