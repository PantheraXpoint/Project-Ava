from AVA.ava import AVA
import argparse
from dataset.init_dataset import init_dataset
from llms.init_model import init_model
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the LLM model to use")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--video_id", type=int, help="ID of the video to process")
    parser.add_argument("--video_path", type=str, help="ID of the video to process")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    dataset = init_dataset(args.dataset)
    llm = init_model(args.model, args.gpus)
    
    if args.video_id is not None:
        video = dataset.get_video(args.video_id)
    else:
        video = dataset.get_video(args.video_path)
    
    start_time = time.time()
    ava = AVA(
        video=video,
        llm_model=llm,
    )
    ava.construct()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    