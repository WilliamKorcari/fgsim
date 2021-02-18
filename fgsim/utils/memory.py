import gc
import os
import sys
from pprint import pprint

import torch


def memReport():

    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            size = obj.view(-1).size()[0]
        else:
            size = sys.getsizeof(obj)
        if size > 1e8:
            if hasattr(obj, "__str__"):
                name = obj.__str__
            elif "__str__" in dir(obj):
                name = obj.__str__()
            if hasattr(obj, "name"):
                name = obj.name
            else:
                name = "Unknown"
            if torch.is_tensor(obj) or type(obj) == dict or len(name) > 20:
                logger.info(f"Type {type(obj)} {size*0.000001}MB")
                pprint(obj)
            else:
                logger.info(f"{name}\t {size*0.000001}MB")


def memGB():
    import psutil

    # logger.debug(psutil.cpu_percent())
    # logger.debug(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    # logger.debug("memory GB:", memoryUse)
    return memoryUse
