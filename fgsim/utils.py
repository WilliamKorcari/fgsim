from collections import OrderedDict
import os
import sys
from .config import conf


def most_freq_zval(posD: OrderedDict):
    from collections import Counter

    cnt = Counter(posD[conf.mapper.xyz[-1]])
    zval = cnt.most_common(1)[0][0]
    return zval


def count_parameters(model):
    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def memReport():
    from pprint import pprint
    import gc
    import torch

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
                print(f"Type {type(obj)} {size*0.000001}MB")
                pprint(obj)
            else:
                print(f"{name}\t {size*0.000001}MB")


def memGB():
    import psutil

    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    # print("memory GB:", memoryUse)
    return memoryUse
