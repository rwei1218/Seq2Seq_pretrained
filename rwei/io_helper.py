import random
import time
# import logging

def data_sample(input_path, out_path, num):
    input_data = list(open(input_path, 'r', encoding='utf8').readlines())
    sample_data = random.sample(input_data, num)
    with open(out_path, 'w', encoding='utf8') as wtf:
        for line in sample_data:
            wtf.write(line)
    print('data sampling ended !')
