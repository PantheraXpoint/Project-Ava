from AVA.ava import AVA
import argparse
from dataset.init_dataset import init_dataset
from llms.init_model import init_model
import time
from AVA.events import extract_events

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the LLM model to use")
    parser.add_argument("--dataset", default="single_video", help="Name of the dataset")
    parser.add_argument("--video_path", type=str, help="ID of the video to process")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    dataset = init_dataset(args.dataset) if args.video_path is None else init_dataset(args.dataset, args.video_path)
    llm = init_model(args.model, args.gpus)

    video = dataset.get_video()
    
    start_time = time.time()
    ava = AVA(
        video=video,
        llm_model=llm,
    )
    events_start_time = time.time()
    events = extract_events(
        llm=ava.llm_model,
        video=ava.global_config["video"],
        global_config=ava.global_config,
    )
    events_end_time = time.time()
    print(f"Time taken for events: {events_end_time - events_start_time} seconds")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    