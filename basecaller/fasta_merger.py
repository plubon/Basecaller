import os
import sys


def merge_fasta(input_path, output_path, format=None):
    files = os.listdir(input_path)
    if format is not None:
        files = [x for x in files if format in x]
    with open(output_path, 'w') as out:
        for file in files:
            with open(os.path.join(input_path, file)) as in_file:
                seq = in_file.read().strip()
                out.write(f">{file}\n")
                out.write(seq)
                out.write('\n')


if __name__ == "__main__":
    merge_fasta(sys.argv[1], sys.argv[2], sys.argv[3])







