import os
import json

def get_helm_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def get_ragtruth_data(path_src, path_resp):
    sources = {}
    responces = {}
    with open(path_src) as f:
        for line in f:
            data = json.loads(line)
            sources[data["source_id"]] = data
    with open(path_resp) as f:
        for line in f:
            data = json.loads(line)
            responces[data["id"]] = data
    return sources, responces