# coding=utf-8

import os
import pickle

def read_cache(cache_path, func):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        data = func()
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        return data
