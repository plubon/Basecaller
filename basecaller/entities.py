
class TrainingExample:

    def __init__(self, id, filename, index, signal, raw_signal, sequence, breaks):
        self.id = id
        self.filename = filename
        self.index = index
        self.signal = signal
        self.sequence = sequence
        self.breaks = breaks
        self.raw_signal = raw_signal

