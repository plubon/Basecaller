import numpy as np
from utils import alphabet_dict
import difflib


class AssemblerFactory:

    @staticmethod
    def get(name):
        if name.lower() == 'simple':
            return SimpleAssembler()


class SimpleAssembler:

    def assemble(self, file_reads):
        consensus = np.zeros([4, 1000])
        pos = 0
        length = 0
        census_len = 1000
        for indx, bpread in enumerate(file_reads):
            if indx == 0:
                self.add_count(consensus, 0, bpread)
                continue
            if np.any(bpread == -1):
                bpread = bpread[:np.argmax(-1)]
            d = difflib.SequenceMatcher(None, file_reads[indx - 1], bpread)
            match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
            disp = match_block[0] - match_block[1]
            if disp + pos + len(file_reads[indx]) > census_len:
                consensus = np.lib.pad(consensus, ((0, 0), (0, 1000)), mode='constant', constant_values=0)
                census_len += 1000
            self.add_count(consensus, pos+disp, file_reads[indx])
            pos += disp
            length = max(length, pos+len(file_reads[indx]))
        return consensus[:, :length]

    def add_count(self, consensus, start_indx, segment):
        if start_indx < 0:
            segment = segment[-start_indx, :]
            start_indx = 0
        for i, base in enumerate(segment):
            consensus[base][start_indx + 1] += 1
