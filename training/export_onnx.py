"""Export the trained YOLO models (.pt) to ONNX for the app.

The app (factory_safety_monitor.py) runs its models with ONNX Runtime, so
after training a model with train_models.py, export it here:

    python export_onnx.py                 # every model listed below
    python export_onnx.py best_helmet.pt  # just one

Run it with the Python that has ultralytics + PyTorch (the training one),
not build_env. Each model is exported at the image size it was TRAINED at
(best_ppe.pt was trained at 512, the rest at 640) with a dynamic input
size, so the app can feed it exactly what ultralytics would - the results
match the .pt model box for box.
"""
import os, sys

# ultralytics otherwise tries to pip-install packages (e.g. onnxruntime-gpu)
# into this Python on its own - never let it change the environment.
os.environ["YOLO_AUTOINSTALL"] = "False"
os.environ["YOLO_OFFLINE"] = "True"

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["yolov8n.pt", "best_ppe.pt", "best_ppe_v2.pt", "best_ppe_plus.pt", "best_smoking.pt",
          "best_helmet.pt", "best_labcoat.pt", "best_gloves_hf.pt"]

def export(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        print(f"skip {name}: not found"); return
    model = YOLO(path)
    imgsz = model.overrides.get("imgsz", 640)            # its training size
    out = model.export(format="onnx", imgsz=imgsz, opset=17, dynamic=True, simplify=False, verbose=False)
    dst = os.path.join(ROOT, os.path.splitext(name)[0] + ".onnx")
    if os.path.abspath(out) != os.path.abspath(dst):
        os.replace(out, dst)
    print(f"OK  {name} -> {os.path.basename(dst)}  (imgsz {imgsz})")

if __name__ == "__main__":
    for n in (sys.argv[1:] or MODELS):
        export(n)
