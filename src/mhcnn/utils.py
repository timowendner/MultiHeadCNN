import torch


class Result:
    def __init__(self, classes: int) -> None:
        self.classes = classes
        self.runs = []

    def register_new(self):
        matrix = torch.zeros((self.classes, self.classes))
        self.runs.append({'n': 0, 'correct': 0, 'matrix': matrix})

    def add(self, outputs: torch.Tensor, targets: torch.Tensor):
        outputs = torch.argmax(outputs, dim=1).cpu()
        targets = torch.argmax(targets, dim=1).cpu()

        run = self.runs[-1]
        run['n'] += outputs.shape[0]
        run['correct'] += torch.sum(outputs == targets)

        run['matrix'][outputs, targets] += 1

    def acc(self, run: int = None):
        if run is None:
            run = -1

        n = self.runs[run]['n']
        correct = self.runs[run]['correct']
        if correct == 0:
            return 0
        return correct / n

    def matrix(self, run: int = None):
        if run is None:
            run = -1

        return self.runs[run]['matrix']
