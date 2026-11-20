# Building with the updated (onedir) spec file

## What changed and why
Your previous build was "onefile" mode - one .exe that unzips itself into a
temp folder on every launch. That's why startup was slow every time, not
just the first time. This new spec uses "onedir" mode instead: everything
gets extracted once, at build time, into a folder. After that, launching is
close to instant.

## 1. Edit the three paths in the .spec file
Open `factory_safety_monitor.spec` and find the `datas=[]` list. Update all
three paths to match where your files actually are:
```python
datas=[
    (r'C:\Users\josef\Desktop\Factory_Safety_System\best_ppe.pt', '.'),
    (r'C:\Users\josef\Desktop\Factory_Safety_System\best_workshop.pt', '.'),
    (r'C:\Users\josef\Desktop\Factory_Safety_System\icon.ico', '.'),
],
```
Also update the `icon=` line near the bottom to the same icon.ico path.

## 2. Build
```
python -m PyInstaller factory_safety_monitor.spec
```

## 3. What you get this time
Instead of a single .exe, you'll see:
```
dist/
  FactorySafetyMonitor/
    FactorySafetyMonitor.exe   <- this is what you run
    icon.ico
    best_ppe.pt
    best_workshop.pt
    _internal/                 <- all the library files live here
      (hundreds of files - torch, opencv, etc.)
```

## 4. This changes how you share it
Before: you could just hand someone one .exe file.
Now: you need to zip the entire FactorySafetyMonitor folder and send that.
The .exe will not work on its own if separated from _internal/ - it
depends on those files sitting right next to it.

To share: right-click the FactorySafetyMonitor folder -> "Compress to
ZIP" -> send that zip. The other person unzips it anywhere and runs the
.exe inside.

## 5. Why this tradeoff is worth it
A single small EXE feels cleaner to hand someone, but "feels cleaner" isn't
worth a slow launch every single time the program runs. Onedir is the
standard recommendation for exactly this kind of app (GUI + heavy ML
libraries) - PyInstaller's own docs steer people toward onedir for anything
beyond a small, simple script.

If launch speed stops mattering to you and you want the single-file feel
back, you can always rebuild with the old onefile-style spec - nothing
about your actual Python code needs to change either way; only the .spec
file's build instructions differ.
