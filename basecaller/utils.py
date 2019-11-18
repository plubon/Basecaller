import json
import os
import re


def get_first_free_device():
    ns = os.popen('nvidia-smi')
    lines = ns.readlines()
    device_lines = []
    for idx, line in enumerate(lines):
        if 'GeForce' in line:
            device_lines.append((idx, line))
    for device_line in device_lines:
        device_id = device_line[1][4]
        split = lines[device_line[0]+1].split('|')
        mem_use = re.findall(r'\d+', split[2])
        if float(mem_use[0])/float(mem_use[1]) < 0.3:
            return device_id
    return None


alphabet_dict = {
	b'A':0,
	b'a':0,
	b'C':1,
	b'c':1,
	b'G':2,
	b'g':2,
	b'T':3,
	b't':3,
    'A':0,
	'a':0,
	'C':1,
	'c':1,
	'G':2,
	'g':2,
	'T':3,
	't':3
	}

int_to_string_label_arr = ['A', 'C', 'G', 'T']

def string_label_to_int(label):
	return [alphabet_dict[x] for x in label]

def int_label_to_string(label):
	return ''.join([int_to_string_label_arr[x] for x in label])

def write_lines_to_file(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write("%s\n" % line)

def write_dict_to_file(path, params):
	with open(path, "w") as json_file:
		json.dump(params, json_file)
		json_file.write("\n")

def log_to_file(path, line):
    with open(path, 'a+') as log_file:
        log_file.write(line)
        log_file.write('\n')