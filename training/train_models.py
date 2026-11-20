"""Train Safety Cam's extra detectors on this PC's GPU.

    python train_models.py ppe        datasets/safety_ppe -> best_ppe_plus.pt
    python train_models.py smoking    datasets/smoking    -> best_smoking.pt
    python train_models.py helmet     datasets/helmet     -> best_helmet.pt
    python train_models.py gloves     datasets/gloves_merged  -> best_gloves.pt   (build with merge_datasets.py)
    python train_models.py goggles    datasets/goggles_merged -> best_goggles.pt
    python train_models.py ppe --resume     continue an interrupted run

Each dataset folder is a Roboflow export in "YOLOv8" format, unzipped (it
must contain a data.yaml somewhere inside). When training finishes, the best
weights are copied next to factory_safety_monitor.py under the name the app
looks for, and the app picks them up on its next start.
"""
import os, sys, glob, shutil, time, yaml

# ultralytics otherwise pip-installs packages into this Python by itself
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS = {
    "ppe":     {"folder": "safety_ppe", "out": "best_ppe_plus.pt"},
    "smoking": {"folder": "smoking",    "out": "best_smoking.pt"},
    "helmet":  {"folder": "helmet",     "out": "best_helmet.pt"},
    "labcoat": {"folder": "labcoat",    "out": "best_labcoat.pt"},
    # merged from several datasets (merge_datasets.py) - big, so fewer epochs
    "gloves":  {"folder": "gloves_merged",  "out": "best_gloves.pt",  "epochs": 40, "patience": 10},
    "goggles": {"folder": "goggles_merged", "out": "best_goggles.pt", "epochs": 30, "patience": 8},
}
BASE_MODEL = "yolov8s.pt"   # small objects (goggles, cigarettes) need more than yolov8n
EPOCHS, PATIENCE, IMGSZ, BATCH, WORKERS = 80, 20, 640, 8, 2


def find_yaml(folder):
    hits = glob.glob(os.path.join(folder, "**", "data.yaml"), recursive=True)
    if not hits:
        sys.exit(f"No data.yaml under {folder} - download the dataset in YOLOv8 format and unzip it there.")
    return hits[0]


def fixed_yaml(src):
    """Roboflow exports use '../train/images'-style paths that don't resolve
    from where ultralytics looks; rewrite them as absolute paths."""
    base = os.path.dirname(src)
    with open(src, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for split in ("train", "val", "test"):
        p = d.get(split)
        if not p: continue
        cands = [os.path.normpath(os.path.join(base, p)),
                 os.path.join(base, p.replace("../", "").replace("..\\", ""))]
        found = next((c for c in cands if os.path.isdir(c)), None)
        if found: d[split] = found
        elif split != "test": sys.exit(f"Can't find the {split} images for {src} (looked in {cands})")
        else: d.pop(split)
    d.pop("path", None)
    out = os.path.join(base, "data_fixed.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True)
    return out, d


def app_class_map():
    """CLASS_ITEM_MAP read straight from the app's source - importing the app
    would also apply its runtime GPU settings to this training process."""
    import ast
    src = open(os.path.join(ROOT, "factory_safety_monitor.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CLASS_ITEM_MAP":
            return ast.literal_eval(node.value)
    sys.exit("CLASS_ITEM_MAP not found in factory_safety_monitor.py")


def check_classes(names):
    """Warn before spending hours training if the app wouldn't understand
    any of this dataset's class names."""
    cmap = app_class_map()
    norm = lambda n: str(n).strip().lower().replace("-", "_").replace(" ", "_")
    names = list(names.values()) if isinstance(names, dict) else list(names)
    mapped = {n: cmap.get(norm(n)) for n in names}
    print("Classes -> app checks:")
    for n, hit in mapped.items():
        print(f"   {n:28s} -> {hit[0] + (' (present)' if hit[1] else ' (missing)') if hit else '(ignored)'}")
    if not any(mapped.values()):
        sys.exit("None of these classes map to a Safety Cam check - add them to CLASS_ITEM_MAP first.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in JOBS:
        sys.exit(__doc__)
    job = JOBS[sys.argv[1]]
    resume = "--resume" in sys.argv
    if sys.platform == "win32":
        # Ask Windows not to idle-sleep while this process runs (same thing a
        # video player does). Changes no settings; ends when training exits.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    from ultralytics import YOLO
    run_name = f"train_{sys.argv[1]}"
    project = os.path.join(ROOT, "runs", "detect")

    if resume:
        last = os.path.join(project, run_name, "weights", "last.pt")
        if not os.path.exists(last): sys.exit(f"Nothing to resume at {last}")
        model = YOLO(last)
        results = model.train(resume=True)
    else:
        data_yaml, d = fixed_yaml(find_yaml(os.path.join(ROOT, "datasets", job["folder"])))
        check_classes(d["names"])
        model = YOLO(BASE_MODEL)
        t0 = time.time()
        # amp=False: GTX 16xx cards (this PC's 1660 Ti) are known to produce
        # NaN losses with mixed precision, and they have no tensor cores to
        # gain from it anyway.
        results = model.train(data=data_yaml, epochs=job.get("epochs", EPOCHS),
                              patience=job.get("patience", PATIENCE), imgsz=IMGSZ,
                              batch=BATCH, workers=WORKERS, amp=False, device=0,
                              project=project, name=run_name, exist_ok=True)
        print(f"Training took {(time.time()-t0)/3600:.1f} h")

    best = os.path.join(project, run_name, "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, os.path.join(ROOT, job["out"]))
        print(f"Saved {job['out']} - restart Safety Cam to use it.")


if __name__ == "__main__":   # required on Windows: dataloader workers re-import this file
    main()
