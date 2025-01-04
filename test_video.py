from ultralytics import YOLO
from video_maker import predict_in_video

def create_a_predicition_test_video(): 
    model = YOLO('runs/detect/train/weights/best.pt')

    input_video_path = "road_signals_dataset/video.mp4"
    output_video_path = "road_signals_dataset/preds_video.mp4"

    predict_in_video(model, input_video_path, output_video_path)



if __name__ == '__main__': 
    create_a_predicition_test_video()