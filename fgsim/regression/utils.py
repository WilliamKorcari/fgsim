import torch
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, path = '.'):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.best_model = None
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
    def __call__(self, model_state, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = model_state
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_model = model_state
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                torch.save(self.best_model, self.path)


def sumInputs(inputs):
    out = []
    for data in inputs:
        out.append(torch.sum(data.x[:, 0]))
    return torch.tensor(out).view((len(out), 1))
