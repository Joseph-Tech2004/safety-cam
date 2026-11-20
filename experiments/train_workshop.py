# -*- coding: utf-8 -*-
from ultralytics import YOLO

def main():
    # Load the base model
    model = YOLO("yolov8n.pt")

    # Added the (1) to match your folder name exactly
    model.train(
        data=r"C:\Users\josef\Desktop\Factory_Safety_System\Oscoscope and toolbox.v1i.yolov8(1)\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,       
        workers=2       
    )

if __name__ == "__main__":
    main()