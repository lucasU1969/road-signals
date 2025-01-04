from ultralytics import YOLO
import cv2

def predict_in_video(model, in_path:str, out_path:str) -> None: 
    cap = cv2.VideoCapture(in_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret: 
            break
        results = model.predict(source=frame, save=False, conf=0.25)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
