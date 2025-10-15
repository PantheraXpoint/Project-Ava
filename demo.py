import gradio as gr
import json
import time

# Load JSON descriptions
with open("/home/nguyendpk/Project-AVAS/AVA_cache/single_video/child_fall_1.mp4/kg/events/events.json", "r") as f:
    descriptions = json.load(f)

# Lookup dict
description_dict = {int(entry["duration"][1]): entry["description"] for entry in descriptions}

def stream_descriptions():
    current_sec = 0
    accumulated = ""  # store all descriptions
    while True:
        time.sleep(1)
        current_sec += 1
        desc = description_dict.get(current_sec, None)
        if desc:
            # Highlight keywords
            desc = desc.replace("fall", "<span style='color:red; font-weight:bold;'>fall</span>")
            desc = desc.replace("ALARMING", "<span style='color:red; font-weight:bold;'>ALARMING</span>")
            
            # Add description with horizontal line separator
            accumulated += f"<p><b>[{current_sec}s]</b> {desc}</p><hr>"
            yield accumulated
        else:
            yield accumulated

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            desc_box = gr.HTML(label="Notifications")   # use HTML for styling
        with gr.Column(scale=2):
            video = gr.Video(
                value="../TrackandLog/YOLO-World/onnx_outputs/output_child_fall_1.mp4",
                # value="datas/single_video/child_fall_1.mp4",
                label="Video",
                autoplay=True,
                loop=True
            )

    demo.load(stream_descriptions, outputs=desc_box)

demo.launch()
