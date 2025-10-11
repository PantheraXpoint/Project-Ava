import time
import os
import json
import re
import numpy as np
from json_repair import repair_json
from typing import Union
from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from llms.BaseModel import BaseVideoModel, BaseLanguageModel
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
)
from .prompt import PROMPTS
from .events import extract_events
from .entities import extract_entities_and_relations
import time


def extract_knowledge_graph(
        llm: BaseVideoModel,
        embedding_model: BaseEmbeddingModel,
        knowledge_graph_inst: BaseGraphStorage,
        events_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        features_vdb: BaseVectorStorage,
        global_config: dict,
        if_check: bool = True
)->Union[BaseGraphStorage]:
    
    if events_vdb.is_empty() or if_check:
        events_start_time = time.time()
        events = extract_events(
            llm=llm,
            video=global_config["video"],
            global_config=global_config,
        )
        events_end_time = time.time()
        print(f"Time taken for events: {events_end_time - events_start_time} seconds")
    else:
        events = events_vdb.get_datas()

    # add to event_vdb
    datas_for_vdb = {
        event["id"]:{
            "id": event["id"],
            "description": event["description"],
            "duration": event["duration"],
            "content": [event["description"]]
        } for event in events
    }
    
    events_vdb.upsert(datas_for_vdb)
    
    # add to features vdb
    if features_vdb.is_empty():
        video = global_config["video"]
        datas_for_vdb = {}
        frame_path = video.frames_dir
        for event in events:
            duration = event["duration"]
            for i in range(int(duration[0]), int(duration[1])):
                _, _, index = video.get_frames_by_fps(fps=1, duration=(i,i))
                if not index:
                    continue
                frame_dir = os.path.join(frame_path, f"{index[0]}.jpg")
                datas_for_vdb[f"{i}"] = {
                    "content": [frame_dir],
                    "event": event["id"],
                    "frame_dir": frame_dir
                }
        
        features_vdb.upsert(datas_for_vdb)
    
    for event in events:
        if not knowledge_graph_inst.has_node(event["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=event["id"],
                node_data={
                    "type": "event",
                    "id": event["id"],
                    "description": event["description"],
                }
            )
    