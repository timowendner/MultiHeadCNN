import torch


class Result:
    def __init__(self, classes: int) -> None:
        self.classes = classes
        self.train = []
        self.test = []
        self.active = self.train

    def register_test(self):
        matrix = torch.zeros((self.classes, self.classes))
        self.test.append({'n': 0, 'correct': 0, 'matrix': matrix})
        self.active = self.test

    def register_train(self):
        matrix = torch.zeros((self.classes, self.classes))
        self.train.append({'n': 0, 'correct': 0, 'matrix': matrix})
        self.active = self.train

    def add(self, outputs: torch.Tensor, targets: torch.Tensor):
        run = self.active[-1]
        outputs = torch.argmax(outputs, dim=1).cpu()
        targets = torch.argmax(targets, dim=1).cpu()

        run['n'] += outputs.shape[0]
        run['correct'] += torch.sum(outputs == targets)
        run['matrix'][outputs, targets] += 1

    def acc_train(self, run: int = None):
        if run is None:
            run = -1

        n = self.train[run]['n']
        correct = self.train[run]['correct']
        if correct == 0:
            return 0
        return correct / n

    def acc_test(self, run: int = None):
        if run is None:
            run = -1

        n = self.test[run]['n']
        correct = self.test[run]['correct']
        if correct == 0:
            return 0
        return correct / n

    def matrix(self, run: int = None):
        if run is None:
            run = -1

        return self.active[run]['matrix']
