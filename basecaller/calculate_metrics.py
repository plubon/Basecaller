from fasta_merger import merge_fasta
import json
import os
import sys

def calculate(fasta_path, out_path):
    with open('metrics_settings.json', 'r') as json_file:
        json_string = json_file.read()
        json_data = json.loads(json_string)
    merge_fasta(fasta_path, os.path.join(out_path, 'merged.fasta'))
    os.system(f"{json_data['minimap']} -a {json_data['reference']} {os.path.join(out_path, 'merged.fasta')} > {os.path.join(out_path, 'merged.sam')}")
    os.system(f"{json_data['samtools']} view -S -b {os.path.join(out_path, 'merged.sam')} > {os.path.join(out_path, 'merged.bam')}")
    os.system(f"{json_data['hts']} --bamFile={os.path.join(out_path, 'merged.bam')} --reference={json_data['reference']} > {os.path.join(out_path, 'metrics.outs')}")


if __name__ == "__main__":
    calculate(sys.argv[1], sys.argv[2])