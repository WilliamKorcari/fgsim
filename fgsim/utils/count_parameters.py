from ..utils.logger import logger
def count_parameters(model):
    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    table.align["Parameters"] = "l"
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params
