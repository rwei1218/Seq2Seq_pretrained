import os
import fnmatch
import sys
import re
import argparse


reg_digit = r"(\+)?\d+(\.\d+)?"
pull_words= ['"', "'", "''", "!", "=", "-",
             "--", ",", "?", ".", ";", ":",
             "``", "`", "-rrb-", "-llb-", "\\/"]
bad_words = ['update#', 'update', 'recasts', 'undated', 'grafs', 'corrects',
             'retransmitting', 'updates', 'dateline', 'writethru',
             'recaps', 'inserts', 'incorporates', 'adv##',
             'ld-writethru', 'djlfx', 'edits', 'byline',
             'repetition', 'background', 'thruout', 'quotes',
             'attention', 'ny###', 'overline', 'embargoed', 'ap', 'gmt',
             'adds', 'embargo',
             'urgent', '?', ' i ', ' : ', ' - ', ' by ', '-lrb-', '-rrb-']

# EVALUTION_PATH = sys.argv[3]
# REFERENCE_PATH = sys.argv[2]
# MODEL_OUTPUT_PATH = sys.argv[1]

def punctuation_pull(sentence):
    words = (sentence.strip()).split(' ')
    new_lines = ''
    for word in words:
        if word not in pull_words:
            new_lines += word + ' '
    return new_lines.strip()

def generate(EVALUTION_PATH, REFERENCE_PATH, MODEL_OUTPUT_PATH):
    appeared = set()
    if not os.path.exists(EVALUTION_PATH):
        os.makedirs(EVALUTION_PATH)
    out_path = os.path.join(EVALUTION_PATH, 'output')
    ref_path = os.path.join(EVALUTION_PATH, 'ref')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(ref_path):
        os.makedirs(ref_path)
    all_xml = '<ROUGE-EVAL version="1.5.5">\n'
    index = 1
    ref = ""
    oup = ""
    ALLID = 1
    ROUGE_PATH = os.path.join(EVALUTION_PATH, 'ROUGE.xml')
    with open(ROUGE_PATH, 'w') as wtf:
        wtf.write(all_xml)

    with open(MODEL_OUTPUT_PATH) as f:
        for line in f:
            items = line.split('\t')
            if len(items) != 2:
                print('ecpect id and output, but only find one item', line)
                continue
            id, output = int(items[0]), items[1].strip()
            appeared.add(id)
            file_name = '%s.html' % str(id)
            dec_html = '<html>\n' \
                       '<head>\n' \
                       '<title>%s</title>\n' \
                       '</head>\n' \
                       '<body bgcolor="white">\n' \
                       '<a name="1">[1]</a> <a href="#1" id=1>%s</a>\n' \
                       '</body>\n' \
                       '</html>\n' % (str(id), punctuation_pull(output))
            with open(os.path.join(out_path, file_name), 'w+') as wtf:
                wtf.write(dec_html)
    with open(REFERENCE_PATH) as f:
        for line in f:
            items = line.split('\t')
            # if len(items) != 2:
            #    print('ecpect id and output, but only find one item', line)
            #    continue
            id = int(items[0])
            dec_file_name = '%s.html' % str(id)
            if id not in appeared: continue
            all_ref_path = []
            flag = False
            ref_sentence = items[1].strip().lower()
            ref_sentence = re.sub(reg_digit, '#', ref_sentence)
            ref_html = '<html>\n' \
                    '<head>\n' \
                    '<title>%s</title>\n' \
                    '</head>\n' \
                    '<body bgcolor="white">\n' \
                    '<a name="1">[1]</a> <a href="#1" id=1>%s</a>\n' \
                    '</body>\n' \
                    '</html>\n' % (str(id), punctuation_pull(ref_sentence))
            out_file_name = '%s.html' % (str(id))
            all_ref_path.append(out_file_name)
            with open(os.path.join(ref_path, out_file_name), 'w+') as wtf:
                wtf.write(ref_html)
            
            xml = '<EVAL ID="%d">\n' \
                  '<PEER-ROOT>\n' \
                  '%s\n' \
                  '</PEER-ROOT>\n' \
                  '<MODEL-ROOT>\n' \
                  '%s\n' \
                  '</MODEL-ROOT>\n' \
                  '<INPUT-FORMAT TYPE="SEE">\n' \
                  '</INPUT-FORMAT>\n' \
                  '<PEERS>\n' \
                  '<P ID="1">%s</P>\n' \
                  '</PEERS>\n' \
                  '<MODELS>\n' \
                  '<M ID="A">%s</M>\n' \
                  '</MODELS>\n' \
                  '</EVAL>\n' % (index, out_path, ref_path, dec_file_name,
                                 all_ref_path[0])
            with open(ROUGE_PATH, 'a') as wtf:
                wtf.write(xml)
            index += 1
    with open(ROUGE_PATH, 'a') as wtf:
        wtf.write('</ROUGE-EVAL>')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_path",
                        default=None,
                        type=str,
                        required=True)

    parser.add_argument("--ref_path",
                        default=None,
                        type=str,
                        required=True)
    
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True)
    
    args = parser.parse_args()

    EVALUTION_PATH = '/root/eval'
    REFERENCE_PATH = '/root/eval_data/ref/gigavalid_ref.txt'
    MODEL_OUTPUT_PATH = '/root/eval_data/baseline/giga_valid_sample500_cp6_summ.txt'

    generate(
        EVALUTION_PATH=args.eval_path,
        REFERENCE_PATH=args.ref_path,
        MODEL_OUTPUT_PATH=args.output_path
    )
    print('rouge xml created !')                
