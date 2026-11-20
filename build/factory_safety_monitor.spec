# factory_safety_monitor.spec
# Build with the light build environment (no PyTorch in it), e.g.:
#   build_env\Scripts\python.exe -m PyInstaller factory_safety_monitor.spec --noconfirm
# (or just run build_exe.bat). The app runs its models with ONNX Runtime +
# DirectML, so the built folder is a fraction of the old 4.4 GB torch build.
#
# WHY THIS IS "onedir" MODE, NOT "onefile"
# -----------------------------------------
# A onefile EXE looks like one tidy file, but every single launch it has to
# silently unzip its ENTIRE contents (onnxruntime, opencv, numpy, all the
# model files, etc) into a fresh
# temp folder before Python can even start. That unzip is what makes a
# onefile build slow on EVERY run, not just the first - there's no "first
# run is slow, then it's fast" with onefile; it's slow every time, because
# the unzip happens every time.
#
# Onedir mode extracts everything ONCE at build time instead. The tradeoff:
# instead of a single FactorySafetyMonitor.exe, you get a folder full of
# files (the exe + a _internal folder with all the libraries) that all need
# to travel together when you share the program. Launch speed afterward is
# close to running the .py directly - no per-launch unzip step.

import os

block_cipher = None

ROOT = r'C:\Users\josef\Desktop\Factory_Safety_System'
# The ONNX exports of the models (made by export_onnx.py). Bundled only if
# they exist, so a model that hasn't been trained/exported yet doesn't
# break the build. Keep in sync with DETECTOR_MODEL_FILES in the .py file.
MODEL_FILES = ['yolov8n.onnx', 'best_ppe.onnx', 'best_ppe_v2.onnx', 'best_ppe_plus.onnx', 'best_smoking.onnx',
               'best_helmet.onnx', 'best_labcoat.onnx', 'best_gloves.onnx']
MODEL_DATAS = [(os.path.join(ROOT, f), '.') for f in MODEL_FILES if os.path.exists(os.path.join(ROOT, f))]

a = Analysis(
    ['factory_safety_monitor.py'],
    pathex=[],
    binaries=[],
    datas=MODEL_DATAS + [
        # Bundled inside dist/FactorySafetyMonitor/_internal/. The right-hand
        # '.' means "place at the root of that internal folder" - this
        # matches where the script's BASE_DIR-relative paths expect to find
        # them. Do not change the right side.
        # Icon bundled as a DATA file too, so the script can find it at
        # runtime via BASE_DIR/icon.ico for the window/taskbar icon (this
        # is separate from the icon='...' line below, which only sets the
        # EXE FILE's own icon as seen in File Explorer - you need both).
        (r'C:\Users\josef\Desktop\Factory_Safety_System\icon.ico', '.'),
        # cloudflared.exe: find_cloudflared() looks for this next to the
        # running app first - without bundling it, remote access would only
        # work if cloudflared happened to be on the target machine's PATH.
        (r'C:\Users\josef\Desktop\Factory_Safety_System\cloudflared.exe', '.'),
    ],
    hiddenimports=['cv2', 'onnxruntime'],
    hookspath=[],
    runtime_hooks=[],
    # never pull the training stack into the app, even if it's installed
    excludes=['torch', 'torchvision', 'ultralytics', 'matplotlib', 'pandas', 'scipy'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FactorySafetyMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=r'C:\Users\josef\Desktop\Factory_Safety_System\icon.ico',
)

# This COLLECT step is what makes it onedir instead of onefile.
# It gathers the exe + all binaries + all data files into one output
# folder (dist/FactorySafetyMonitor/) instead of compressing them inside
# the exe. This is the only structural difference from a onefile spec.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FactorySafetyMonitor',
)
