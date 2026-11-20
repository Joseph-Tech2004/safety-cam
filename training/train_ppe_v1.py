from ultralytics import YOLO

if __name__ == '__main__':
    # Load the official pre-trained nano model to start transfer learning
    model = YOLO("yolov8n.pt")

    # Start the training run on your custom dataset
    model.train(
        data="data.yaml",    # Points to your classes config
        epochs=50,           # Total complete dataset run iterations
        imgsz=512,           # Image dimension resolution from Roboflow
        device=0             # Forces the run onto your NVIDIA GPU
    )