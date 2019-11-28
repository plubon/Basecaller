from fasta_merger import merge_fasta
import json
import os
import sys

def calculate(fasta_path, out_path):
    with open('metrics_settings.json', 'r') as json_file:
        json_string = json_file.read()
        json_data = json.loads(json_string)
    merge_fasta(fasta_path, os.path.join(out_path, 'ecoli_merged.fasta'), format='ecoli')
    os.system(f"{json_data['minimap']} -a {json_data['ecoli_reference']} {os.path.join(out_path, 'ecoli_merged.fasta')} > {os.path.join(out_path, 'ecoli_merged.sam')}")
    os.system(f"{json_data['samtools']} view -S -b {os.path.join(out_path, 'ecoli_merged.sam')} > {os.path.join(out_path, 'ecoli_merged.bam')}")
    os.system(f"{json_data['hts']} --bamFile={os.path.join(out_path, 'ecoli_merged.bam')} --reference={json_data['ecoli_reference']} > {os.path.join(out_path, 'ecoli_metrics.outs')}")
    merge_fasta(fasta_path, os.path.join(out_path, 'lambda_merged.fasta'), format='Lambda')
    os.system(f"{json_data['minimap']} -a {json_data['lambda_reference']} {os.path.join(out_path, 'lambda_merged.fasta')} > {os.path.join(out_path, 'lambda_merged.sam')}")
    os.system(f"{json_data['samtools']} view -S -b {os.path.join(out_path, 'lambda_merged.sam')} > {os.path.join(out_path, 'lambda_merged.bam')}")
    os.system(f"{json_data['hts']} --bamFile={os.path.join(out_path, 'lambda_merged.bam')} --reference={json_data['lambda_reference']} > {os.path.join(out_path, 'lambda_metrics.outs')}")


if __name__ == "__main__":
    calculate(sys.argv[1], sys.argv[2])