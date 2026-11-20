"""Build the merged GLOVES and GOGGLES training sets from several datasets.

    python merge_datasets.py            -> datasets/gloves_merged, datasets/goggles_merged

Why merge: every public dataset alone covered one look only (white work
gloves on construction sites, blue nitrile gloves held up to a phone, clear
lab goggles, one welder...). Our workshop needs all of them, so each source's
classes are mapped onto two classes per item - present / missing - and
put together. Images with none of the item's classes are left out.

Sources (all Roboflow / Ultralytics exports, YOLO format):
  gloves_augment     Gloves, No Gloves        - construction, labs, CCTV, bare hands
  gloves_color       gloves                   - blue / white / pink nitrile & latex
  lab_gloves         gloves, goggles          - lab coat + nitrile + lab goggles
  welding_ppe        close_goggle, gloves     - welding goggles + welding gloves
  goggles_lab        goggles                  - clear lab safety goggles
  construction_ppe   gloves / no_gloves, goggles / no_goggle   (Ultralytics)
  safety_ppe         Glove / No_Glove, Goggles / No_Goggles    (what best_ppe_plus learned)
"""
import os, glob, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "datasets")

# (folder, {split name in that dataset: our split}, {its class id: our class id})
GLOVES = [
    ("gloves_augment",   {"train": "train", "valid": "valid"}, {0: 0, 1: 1}),
    ("gloves_color",     {"train": "train", "valid": "valid"}, {0: 0}),
    ("lab_gloves",       {"train": "train", "valid": "valid"}, {0: 0}),
    ("welding_ppe",      {"train": "train", "valid": "valid"}, {2: 0}),
    ("construction_ppe", {"train": "train", "val": "valid"},   {1: 0, 9: 1}),
    ("safety_ppe",       {"train": "train", "valid": "valid"}, {0: 0, 4: 1}),
]
GOGGLES = [
    ("welding_ppe",      {"train": "train", "valid": "valid"}, {1: 0}),          # close_goggle = on the eyes
    ("goggles_lab",      {"train": "train", "valid": "valid"}, {1: 0}),          # its class 0 is helmets
    ("lab_gloves",       {"train": "train", "valid": "valid"}, {1: 0}),
    ("construction_ppe", {"train": "train", "val": "valid"},   {4: 0, 8: 1}),
    ("safety_ppe",       {"train": "train", "valid": "valid"}, {1: 0, 5: 1}),
]

def split_dirs(folder, split):
    """(images dir, labels dir) for both layouts: split/images  or  images/split."""
    for img, lbl in ((os.path.join(folder, split, "images"), os.path.join(folder, split, "labels")),
                     (os.path.join(folder, "images", split), os.path.join(folder, "labels", split))):
        if os.path.isdir(img) and os.path.isdir(lbl): return img, lbl
    return None, None

def to_box(vals):
    """A polygon line (x1 y1 x2 y2 ...) becomes its bounding box (cx cy w h)."""
    if len(vals) == 4: return vals
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

def build(name, sources, class_names):
    out = os.path.join(D, name)
    if os.path.exists(out): shutil.rmtree(out)
    counts = {}
    for folder, splits, cmap in sources:
        for src_split, our_split in splits.items():
            img_dir, lbl_dir = split_dirs(os.path.join(D, folder), src_split)
            if img_dir is None:
                print(f"   (skip {folder}/{src_split}: not found)"); continue
            oi = os.path.join(out, our_split, "images"); ol = os.path.join(out, our_split, "labels")
            os.makedirs(oi, exist_ok=True); os.makedirs(ol, exist_ok=True)
            n = 0
            for lbl in glob.glob(os.path.join(lbl_dir, "*.txt")):
                lines = []
                for ln in open(lbl, encoding="utf-8", errors="ignore"):
                    p = ln.split()
                    if len(p) < 5 or int(float(p[0])) not in cmap: continue
                    box = to_box([float(v) for v in p[1:]])
                    if box[2] <= 0 or box[3] <= 0: continue
                    lines.append(f"{cmap[int(float(p[0]))]} " + " ".join(f"{v:.6f}" for v in box))
                if not lines: continue
                stem = os.path.splitext(os.path.basename(lbl))[0]
                imgs = glob.glob(os.path.join(img_dir, glob.escape(stem) + ".*"))
                if not imgs: continue
                new = f"{folder}__{stem}"                       # keep names unique across sources
                shutil.copy(imgs[0], os.path.join(oi, new + os.path.splitext(imgs[0])[1]))
                with open(os.path.join(ol, new + ".txt"), "w") as f: f.write("\n".join(lines) + "\n")
                n += 1
            counts[f"{folder}/{our_split}"] = n
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"train: {os.path.join(out, 'train', 'images')}\nval: {os.path.join(out, 'valid', 'images')}\n"
                f"nc: {len(class_names)}\nnames: {class_names}\n")
    print(name, counts, "total", sum(counts.values()))

if __name__ == "__main__":
    build("gloves_merged", GLOVES, ["gloves", "no_gloves"])
    build("goggles_merged", GOGGLES, ["goggles", "no_goggles"])
