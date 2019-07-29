from utils import alphabet_dict
from sklearn.model_selection import train_test_split


class H5FileReader:

    def __init__(self, parser):
        self.parser = parser

    def find_signal(self, name):
	    if 'Signal' in name:
	        return name

    def read(self, path):
        with h5py.File(path+'.fast5', 'r') as h5_file:
            if 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' not in h5_file:
                return []
            corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
            corrected_events_array = corrected_events[()]
            raw_signal = h5_file['Raw/Reads']
            signal_path = raw_signal.visit(self.find_signal)
            dataset = raw_signal[signal_path][()]
            offset = corrected_events.attrs.get('read_start_rel_to_raw')
            event_position = np.array([x[2] + offset for x in corrected_events_array])
            event_length = np.array([x[3] for x in corrected_events_array])
            sequence = np.array([x[4] for x in corrected_events_array])
            i = self.parser.skip
            current_start = i
            current_len = 0
            examples = []
            while i < len(event_position) - self.parser.skip:
                if sequence[i] not in alphabet_dict.keys():
                    current_len = 0
                    current_start = i + 1
                    i += 1
                else:
                    if current_len + event_length[i] < self.parser.segment_length:
                        current_len += event_length[i]
                        i += 1 
                    else:
                        if i - current_start >4 :
                            signal = dataset[event_position[current_start]: event_position[current_start]+self.parser.segment_length]
                            normalized_signal = (signal - np.mean(np.unique(dataset)))/np.std(np.unique(dataset))
                            breaks = event_position[current_start+1:i-1] - event_position[current_start]
                            current_seq = sequence[current_start:i-1]
                            example = TrainingExample(self.parser.get_id(), path, current_start, normalized_signal, signal, current_seq, breaks)
                            examples.append(example)
                        current_len = 0
                        current_start = i + 1
                        i += 1
            return examples


class ChironFileReader:

    def __init__(self, parser):
        self.parser = parser

    def read(self, path):
        with open(path+'.label', 'r') as label_file:
            with open(path+'.signal', 'r') as signal_file:
                dataset = signal_file.readlines()
                dataset = dataset[0].strip()
                dataset = dataset.split(' ')
                dataset = [int(x) for x in dataset]
                corrected_events_array = label_file.readlines()
                corrected_events_array = [x.strip() for x in corrected_events_array]
                corrected_events_array = [x.split(' ') for x in corrected_events_array]
                event_position = np.array([int(x[0]) for x in corrected_events_array])
                event_length = np.array([int(x[1]) - int(x[0]) for x in corrected_events_array])
                sequence = np.array([x[2] for x in corrected_events_array])
                i = self.parser.skip
                current_start = i
                current_len = 0
                examples = []
                while i < len(event_position) - self.parser.skip:
                    if sequence[i] not in alphabet_dict.keys():
                        current_len = 0
                        current_start = i + 1
                        i += 1
                    else:
                        if current_len + event_length[i] < self.parser.segment_length:
                            current_len += event_length[i]
                            i += 1 
                        else:
                            if i - current_start >4 :
                                signal = dataset[event_position[current_start]: event_position[current_start]+self.parser.segment_length]
                                normalized_signal = (signal - np.mean(np.unique(dataset)))/np.std(np.unique(dataset))
                                breaks = event_position[current_start+1:i-1] - event_position[current_start]
                                current_seq = sequence[current_start:i-1]
                                example = TrainingExample(self.parser.get_id(), path, current_start, normalized_signal, signal, current_seq, breaks)
                                examples.append(example)
                            current_len = 0
                            current_start = i + 1
                            i += 1
                return examples

class SignalFileParser:

    known_types = {
        'h5': H5FileReader(), 
        'chiron': ChironFileReader()
        }

    def get_id(self):
        id = self.id_counter
        self.id_counter += 1
        return str(id)

    def __init__(input_path, output_path, seg_length=300, skip=10, type='h5', test_size=0.2, val_size=0.2):
        if type not in known_types.keys():
            raise ValueError(f"Type was {type}, but it must be one of {" ".join(known_types)}.")
        self.input_path = input_path
        self.output_path = output_path
        self.files = os.listdir(self.input_path)
        self.files = [x.split('.')[0] for x in self.files]
        self.files = list(set(self.files))
        self.skip = skip
        self.segment_length = seg_length
        self.id_counter = 1
        self.reader = known_types[self.type]
        self.test_size = test_size
        self.val_size = val_size

    def create():
        train, rest = train_test_split(self.files, train_size=1-self.test_size-self.val_size)
        val, test = train_test_split(rest, train_size = self.val_size/(self.val_size+self.train_size))
        
