from collections import OrderedDict

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
